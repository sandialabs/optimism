from scipy.sparse import diags
from matplotlib import pyplot as plt

from optimism.JaxConfig import *
from optimism.phasefield import PhaseFieldThreshold as material
from optimism.phasefield import PhaseField
from optimism.phasefield import MaterialPointSimulator
from optimism.test import MeshFixture
from optimism import Mesh
from optimism import AlSolver
from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Objective
from optimism.ConstrainedObjective import ConstrainedObjective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import TensorMath


E = 1.0
nu = 0.25
Gc = 0.2
ell = 0.25

props = {'elastic modulus': E,
         'poisson ratio': nu,
         'critical energy release rate': Gc,
         'regularization length': ell,
         'kinematics': 'large deformations'}

materialModel = material.create_material_model_functions(props)

subProblemSettings = EqSolver.get_settings()
alSettings = AlSolver.get_settings()

class TestSingleMeshFixture(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 7
        self.Ny = 7
        xRange = [0.,1.]
        yRange = [0.,1.]

        self.targetFieldGrad = np.array([[0.1, -0.2],[0.4, -0.1], [0,0]])
        self.targetFieldOffset = np.array([0.5, -0.5, 0.0]) 

        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x: self.targetFieldGrad@x + self.targetFieldOffset)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        
        ebc = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
               FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        dim = 3 # 2 displacements and scalar phase
        self.dofManager = FunctionSpace.DofManager(self.fs, dim, ebc)
        self.Ubc = self.dofManager.get_bc_values(self.U)

        self.bvpFunctions =  PhaseField.create_phasefield_functions(self.fs,
                                                                    "plane strain",
                                                                    materialModel)

        self.nElements = Mesh.num_elements(self.mesh)
        self.nQuadPtsPerElem = QuadratureRule.len(quadRule)
        self.stateVars = self.bvpFunctions.compute_initial_state()

        dofToUnknown = self.dofManager.dofToUnknown.reshape(self.U.shape)
        self.phaseIds = dofToUnknown[self.dofManager.isUnknown[:,2],2]
        

    def test_sparse_hessian_at_zero_phase(self):
        elementStiffnesses = self.bvpFunctions.\
            compute_element_stiffnesses(self.U, self.stateVars)

        KSparse = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                         self.mesh.conns,
                                                                         self.dofManager)
        KSparseDensified = np.array(KSparse.todense())

        Ubc = self.dofManager.get_bc_values(self.U)
        
        def objective_func(Uu):
            U = self.dofManager.create_field(Uu, Ubc)
            return self.bvpFunctions.compute_internal_energy(U, self.stateVars)

        compute_dense_hessian = hessian(objective_func)
        KDense = compute_dense_hessian(self.dofManager.get_unknown_values(self.U))
        
        self.assertArrayNear(KSparseDensified, KDense, 11)


    def test_sparse_hessian_at_nonzero_phase(self):
        U = self.U.at[:,2].set(np.linspace(0.1, 0.9, self.Nx*self.Ny))

        elementStiffnesses = self.bvpFunctions.\
            compute_element_stiffnesses(U, self.stateVars)

        KSparse = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                         self.mesh.conns,
                                                                         self.dofManager)
        KSparseDensified = np.array(KSparse.todense())

        Ubc = self.dofManager.get_bc_values(U)
        
        def objective_func(Uu):
            U = self.dofManager.create_field(Uu, Ubc)
            return self.bvpFunctions.compute_internal_energy(U, self.stateVars)

        compute_dense_hessian = hessian(objective_func)
        KDense = compute_dense_hessian(self.dofManager.get_unknown_values(U))
        
        self.assertArrayNear(KSparseDensified, KDense, 11)


    def test_constrained_hessian(self):
        U = self.U
        
        c = U[self.dofManager.isUnknown[:,2],2]
        lam = np.zeros_like(c)
        kappa = np.ones_like(lam)

        elementStiffnesses =self.bvpFunctions.\
            compute_element_stiffnesses(U, self.stateVars)

        KSparse = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                         self.mesh.conns,
                                                                         self.dofManager)
        constraintHessian = self.bvpFunctions.compute_constraint_hessian(lam,
                                                                         kappa,
                                                                         c,
                                                                         self.dofManager)

        KSparse += diags(constraintHessian, 0, format='csc')
        KDensified = np.array(KSparse.todense())

        # construct constrained obj
        Ubc = self.dofManager.get_bc_values(U)
        Uu = self.dofManager.get_unknown_values(U)
        
        def objective_function(Uu, p):
            U = self.dofManager.create_field(Uu, Ubc)
            return self.bvpFunctions.compute_internal_energy(U, self.stateVars)

        def constraint_function(Uu, p):
            U = self.dofManager.create_field(Uu, Ubc)    
            return U[self.dofManager.isUnknown[:,2],2]

        objective = ConstrainedObjective(objective_function,
                                         constraint_function,
                                         Uu,
                                         None,
                                         lam,
                                         kappa)
        
        KDense = objective.hessian(Uu)

        self.assertArrayNear(KDensified, KDense, 11)


    def test_patch_test(self):
        
        def objective_function(Uu, p):
            U = self.dofManager.create_field(Uu, self.Ubc)
            return self.bvpFunctions.compute_internal_energy(U, self.stateVars)

        def constraint_function(Uu, p):
            return Uu[self.phaseIds]
                                                  
        Uu_guess = self.dofManager.get_unknown_values(self.U)

        p0 = Objective.Params(None, self.stateVars)
        lam0 = grad(objective_function, 0)(Uu_guess, p0)[self.phaseIds]
        kappa0 = 10.0*np.amax(lam0) * np.ones(lam0.shape)

        objective = ConstrainedObjective(objective_function,
                                         constraint_function,
                                         Uu_guess,
                                         p0, 
                                         lam0, 
                                         kappa0)
           
        Uu = AlSolver.augmented_lagrange_solve(objective, Uu_guess, p0,
                                               alSettings, subProblemSettings,
                                               useWarmStart=False)

        U = self.dofManager.create_field(Uu, self.Ubc)

        fieldGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        fieldGrads = fieldGrads.reshape((self.nElements*self.nQuadPtsPerElem,
                                         fieldGrads.shape[2], fieldGrads.shape[3]))
        for fg in fieldGrads:
             self.assertArrayNear(fg, self.targetFieldGrad, 10)

        lagrangian = lambda x : objective_function(x, 0.0) - constraint_function(x, 0.0) @ objective.lam
        grad_lagrangian = grad(lagrangian)
        self.assertArrayNear( grad_lagrangian(Uu), np.zeros(Uu.shape), 10)

            
if __name__ == '__main__':
    MeshFixture.unittest.main()

