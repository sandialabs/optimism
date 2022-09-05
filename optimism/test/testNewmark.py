import jax
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

        self.dynamicsFunctions = Mechanics.create_dynamics_functions(self.fs,
                                                                     'plane strain',
                                                                     materialModel,
                                                                     newmarkParams)

        self.staticsFunctions = Mechanics.create_mechanics_functions(self.fs,
                                                                     'plane strain',
                                                                     materialModel)
        
        # using an elastic model, so we can neglect internal var updating
        self.internalVariables = self.dynamicsFunctions.compute_initial_state()
        
        EBCs = [Mesh.EssentialBC(nodeSet='bottom', field=1)]
        self.dofManager =  Mesh.DofManager(self.mesh, self.fieldShape, EBCs)

    def test_potential(self):
        key = jax.random.PRNGKey(1)
        U = jax.random.uniform(key, self.mesh.coords.shape)
        key, subkey = jax.random.split(key)
        V = 0.1*jax.random.uniform(subkey, self.mesh.coords.shape)
        key, subkey = jax.random.split(subkey)
        A = jax.random.uniform(subkey, self.mesh.coords.shape)
        dt = 0.1
        UPre, _ = self.dynamicsFunctions.predict(U, V, A, dt)
        action = self.dynamicsFunctions.compute_algorithmic_energy(U, UPre, self.internalVariables, dt)
        print(action)
        self.assertGreater(np.abs(action), 0.0)
        
    def test_hessian_matrix_is_symmetric(self):
        key = jax.random.PRNGKey(1)
        U = jax.random.uniform(key, self.mesh.coords.shape)
        key, subkey = jax.random.split(key)
        V = 0.1*jax.random.uniform(subkey, U.shape)
        key, subkey = jax.random.split(subkey)
        A = jax.random.uniform(subkey, U.shape)
        dt = 0.1
        UPre, _ = self.dynamicsFunctions.predict(U, V, A, dt)
        elementHessians = self.dynamicsFunctions.compute_element_hessians(U, UPre, self.internalVariables, dt)
        K = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementHessians,
                                                                   self.mesh.conns,
                                                                   self.dofManager)
        KSkew = 0.5*(K.todense() - K.todense().T)
        asymmetry = np.linalg.norm(KSkew.ravel(), np.inf)
        self.assertLessEqual(asymmetry, 1e-12)


    def test_compute_kinetic_energy(self):
        velocity = np.array([3.1, 0.7])
        V = np.zeros(self.mesh.coords.shape)
        V = V.at[:,0].set(velocity[0])
        V = V.at[:,1].set(velocity[1])
        T = self.dynamicsFunctions.compute_output_kinetic_energy(V)
        m = self.w*self.L*rho
        TExact = 0.5*m*np.dot(velocity, velocity)
        self.assertNear(T, TExact, 14)


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
            Uexact = np.zeros(self.fieldShape).at[:,0].set(t)
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

        # set constant acceleration of 1.0 in x direction
        A = np.zeros_like(self.mesh.coords).at[:, 0].set(1.0)
        Au = self.dofManager.get_unknown_values(A)

        for i in range(1, 15):
            Uu, Vu, Au = self.time_step(Uu, Vu, Au, objective, dt)
            U = self.dofManager.create_field(Uu, self.get_ubcs())
            t = objective.p.time[0]
            UExact = np.zeros(U.shape).at[:,0].set(t + 0.5*t**2)
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
        V = zeroField.at[:,0].set(v0)
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
