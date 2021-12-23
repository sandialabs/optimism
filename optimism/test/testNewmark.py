import unittest
from scipy.sparse import linalg
import numpy as onp

from optimism.JaxConfig import *
from optimism import EquationSolver
from optimism import FunctionSpace
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

from optimism.material import LinearElastic as Material
from optimism.test import MeshFixture

E = 10.0
nu = 0.0
rho = 1.0

trSettings = EquationSolver.get_settings(max_cg_iters=50,
                                         max_trust_iters=500,
                                         min_tr_size=1e-13,
                                         tol=4e-12,
                                         use_incremental_objective=False)


class DynamicsFixture(MeshFixture.MeshFixture):
    def setUp(self):
        self.w = 0.1
        self.L = 1.0
        N = 3
        M = 2
        xRange = [0.0, self.L]
        yRange = [0.0, self.w]
        mesh, _ = self.create_mesh_and_disp(N, M, xRange, yRange, lambda X: 0*X)
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, createNodeSetsFromSideSets=True)

        quadPrecision = 2*(self.mesh.masterElement.degree - 1)
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=quadPrecision)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        self.fieldShape = self.mesh.coords.shape

        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'density': rho}
        materialModel = Material.create_material_model_functions(props)
        newmarkParams = Mechanics.NewmarkParameters(gamma=0.5, beta=0.25)
        self.elementMasses = Mechanics.compute_element_masses(rho, self.mesh)

        self.dynamicsFunctions = Mechanics.create_dynamics_functions(self.fs,
                                                                     'plane strain',
                                                                     materialModel,
                                                                     newmarkParams,
                                                                     self.elementMasses)

        self.staticsFunctions = Mechanics.create_mechanics_functions(self.fs,
                                                                     'plane strain',
                                                                     materialModel)
        
        # using an elastic model, so we can neglect internal var updating
        self.internalVariables = self.dynamicsFunctions.compute_initial_state()
        
        EBCs = [Mesh.EssentialBC(nodeSet='bottom', field=1)]
        self.dofManager =  Mesh.DofManager(self.mesh, self.fieldShape, EBCs)
      
        
    def test_total_mass_in_mass_matrix_is_correct(self):
        spaceDim = 2
        mass = np.sum(self.elementMasses.ravel())/spaceDim
        massExact = self.w*self.L*rho
        self.assertNear(mass, massExact, 14)

        
    def test_mass_matrix_is_symmetric(self):
        M = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(self.elementMasses,
                                                                   self.mesh.conns,
                                                                   self.dofManager)
        MSkew = 0.5*(M.todense() - M.todense().T)
        asymmetry = np.linalg.norm(MSkew.ravel(), np.inf)
        self.assertLessEqual(asymmetry, 1e-12)


    def test_compute_kinetic_energy(self):
        velocity = 3.1
        V = np.zeros(self.mesh.coords.shape)
        V = ops.index_update(V, ops.index[:,0], velocity)
        T = Mechanics.compute_kinetic_energy(V, self.elementMasses, self.mesh.conns)
        m = self.w*self.L*rho
        TExact = 0.5*m*velocity**2
        self.assertNear(T, TExact, 14)


    def test_sparse_hessian_matches_dense_hessian(self):
        Uu, Vu, Au = self.set_initial_conditions()
        dt = 0.1
        tOld = 0.0
        t = tOld + dt
        p = Objective.Params(None,
                             self.internalVariables,
                             None,
                             None,
                             np.array([t, tOld]),
                             Uu)

        def objective_function(Uu, p):
            U = self.create_field(Uu, p)
            UuPre = p.dynamic_data
            UPre = self.create_field(UuPre, p)
            internalVariables = p[1]
            dt = p.time[0] - p.time[1]
            return self.dynamicsFunctions.compute_algorithmic_energy(U, UPre, internalVariables, dt)

        HDense = hessian(objective_function)(Uu, p)

        eH = self.dynamicsFunctions.compute_element_hessians(self.create_field(Uu, p),
                                                             self.internalVariables,
                                                             dt)
        HSparse = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(eH,
                                                                         self.mesh.conns,
                                                                         self.dofManager)

        HSparse = np.array(HSparse.todense())
        self.assertArrayNear(HSparse, HDense, 13)
        


    def test_integration_of_rigid_motion_is_exact(self):
        Uu, Vu, Au = self.set_initial_conditions()
        dt = 0.75
        t = 0.0
        tOld = -dt
        p = Objective.Params(None,
                             self.internalVariables,
                             None,
                             None,
                             np.array([t, tOld]),
                             Uu)

        
        def objective_function(Uu, p):
            U = self.create_field(Uu, p)
            UuPre = p.dynamic_data
            UPre = self.create_field(UuPre, p)
            internalVariables = p[1]
            dt = p.time[0] - p.time[1]
            return self.dynamicsFunctions.compute_algorithmic_energy(U, UPre, internalVariables, dt)
        
        objective = Objective.Objective(objective_function, Uu, p)

        for i in range(1, 15):
            print('---------------------------')
            print('Time Step ', i)
            Uu, Vu, Au = self.time_step(Uu, Vu, Au, objective, dt)
            U = self.create_field(Uu, p)
            t = objective.p[4][0]
            Uexact = ops.index_update(np.zeros(self.fieldShape), ops.index[:,0], t)
            self.assertArrayNear(U, Uexact, 14)


    def test_integration_of_constant_acceleration_is_exact(self):
        Uu, Vu, _ = self.set_initial_conditions()
        dt = 0.75
        t = 0.0
        tOld = -dt
        p = Objective.Params(None,
                             self.internalVariables,
                             None,
                             None,
                             np.array([t, tOld]),
                             Uu)

        def objective_function(Uu, p):
            U = self.create_field(Uu, p)
            UuPre = p.dynamic_data
            UPre = self.create_field(UuPre, p)
            internalVariables = p[1]
            return self.dynamicsFunctions.compute_algorithmic_energy(U, UPre, internalVariables, dt) \
                + self.constant_body_force_potential(Uu, p)
                
                
        objective = Objective.Objective(objective_function, Uu, p)

        MSparse = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(self.elementMasses, self.mesh.conns, self.dofManager)
        Fu = -grad(self.constant_body_force_potential)(Uu, p)
        Au,_ = linalg.cg(MSparse, Fu, atol=1e-10)
        Au = np.array(Au)

        for i in range(1, 15):
            Uu, Vu, Au = self.time_step(Uu, Vu, Au, objective, dt)
            U = self.dofManager.create_field(Uu, self.get_ubcs())
            t = objective.p.time[0]
            UExact = ops.index_update(np.zeros(U.shape), ops.index[:,0], t + 0.5*t**2)
            self.assertArrayNear(U, UExact, 10)

            
    #        
    # helper functions
    #

    
    def time_step(self, Uu, Vu, Au, objective, dt):
        tOld = objective.p.time[0]
        t = tOld + dt
        print('\ttime = ', t, '\tdt = ', dt)
        objective.p = Objective.param_index_update(objective.p, 4, np.array([t, tOld]))
        
        UuPredicted, Vu = self.dynamicsFunctions.predict(Uu, Vu, Au, dt)
        objective.p = Objective.param_index_update(objective.p, 5, UuPredicted)

        Uu = EquationSolver.nonlinear_equation_solve(objective,
                                                     UuPredicted,
                                                     objective.p,
                                                     trSettings,
                                                     useWarmStart=False)

        UuCorrection = Uu - UuPredicted
        Vu, Au = self.dynamicsFunctions.correct(UuCorrection, Vu, Au, dt)
        return Uu, Vu, Au


    def set_initial_conditions(self):
        zeroField = np.zeros(self.mesh.coords.shape)
        Uu = self.dofManager.get_unknown_values(zeroField)

        v0 = 1.0
        V = ops.index_update(zeroField, ops.index[:,0], v0)
        Vu = self.dofManager.get_unknown_values(V)

        Au = self.dofManager.get_unknown_values(zeroField)
        return Uu, Vu, Au


    def get_ubcs(self):
        Ubc = self.dofManager.get_bc_values(np.zeros(self.mesh.coords.shape))
        #Vbc = self.dofManager.get_bc_values(np.zeros(self.mesh.coords.shape))
        #Abc = self.dofManager.get_bc_values(np.zeros(self.mesh.coords.shape))
        return Ubc

    
    def create_field(self, Uu, p):
        Ubc = self.get_ubcs()
        return self.dofManager.create_field(Uu, Ubc)

    
    def constant_body_force_potential(self, Uu, p):
        U = self.dofManager.create_field(Uu, self.get_ubcs())
        internalVariables = p[1]
        b = np.array([1.0, 0.0])
        f = lambda u, du, q, x: -np.dot(b, u)
        return FunctionSpace.integrate_over_block(self.fs, U, internalVariables,
                                                  f, self.mesh.blocks['block'])


if __name__=="__main__":
    unittest.main()
