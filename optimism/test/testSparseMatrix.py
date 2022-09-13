from scipy.sparse import linalg
from scipy.linalg import eigh

from optimism.JaxConfig import *
from optimism.material import LinearElastic as MaterialModel
from optimism.SparseMatrixAssembler import *
from optimism.test import MeshFixture
from optimism import Mesh
from optimism import Mechanics
from optimism import QuadratureRule
from optimism import FunctionSpace


E = 1.0
nu = 0.25


class SparsePatchFixture(MeshFixture.MeshFixture):

    def setUp(self):
        Nx = 3
        Ny = 3
        xRange = [0.0, 1.0]
        yRange = [0.0, 1.0]
        initial_disp_func = lambda x: 0.0*x
        self.mesh, self.U = self.create_mesh_and_disp(Nx, Ny,
                                                      xRange, yRange,
                                                      initial_disp_func)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)

        props = {'elastic modulus': E,
                 'poisson ratio': nu}
        materialModel = MaterialModel.create_material_model_functions(props)

        mechFuncs = Mechanics.create_mechanics_functions(self.fs, "plane strain", materialModel)

        self.internals = mechFuncs.compute_initial_state()

        self.elementStiffnesses = mechFuncs.compute_element_stiffnesses(np.zeros_like(self.U), self.internals)
        self.compute_energy = mechFuncs.compute_strain_energy


    def test_sparse_matrix_patch_test_noBC(self):
        dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=[])
        K = assemble_sparse_stiffness_matrix(self.elementStiffnesses,
                                             self.mesh.conns,
                                             dofManager)

        symmetryNorm = linalg.norm(K - K.T, onp.inf)
        self.assertLess(symmetryNorm, 1e-14)

        # There should be 3 rigid body modes
        Lambda, V = eigh(K.todense())
        self.assertGreater(onp.amin(Lambda[3:]), 1e-14)
        self.assertLess(onp.amax(Lambda[0:3]), 1e-14)

        # check zero energy modes

        # translation in x
        nNodes = self.U.shape[0]
        mode = onp.zeros((nNodes, 2))
        mode[:,0] = 1.0
        modeEnergy = mode.flatten().T @ K @ mode.flatten()
        # print('Rigid mode 1=', modeEnergy)
        self.assertLess(modeEnergy, 1e-14)

        # translation in y
        mode[:] = 0.0
        mode[:,1] = 0.0
        modeEnergy = mode.flatten().T @ K @ mode.flatten()
        # print('Rigid mode 2=', modeEnergy)
        self.assertLess(modeEnergy, 1e-14)

        # rotation about z
        mode[:] = 0.0
        mode[:,0] = -1.0
        mode[:,1] = 1.0
        modeEnergy = mode.flatten().T @ K @ mode.flatten()
        # print('Rigid mode 3=', modeEnergy)
        self.assertLess(modeEnergy, 1e-14)

        
    def test_sparse_matrix_patch_test_traction_BC(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)
        Ubc = dofManager.get_bc_values(self.U)
        
        K = assemble_sparse_stiffness_matrix(self.elementStiffnesses,
                                             self.mesh.conns,
                                             dofManager)
        
        F = onp.zeros(12)       
        F[1] = 4.0/15.0
        F[5] = 8.0/15.0
        F[10] = 4.0/15.0
        
        dU = linalg.spsolve(K, F)
        UNew = dofManager.create_field(dU, Ubc)

        # plane strain, Uy should be -nu/(1-nu)*Ux
        exactSolution = lambda x: np.column_stack((x[:,0], -nu/(1.0 - nu)*x[:,1]))
        UExact = exactSolution(self.mesh.coords)
        self.assertArrayNear(UNew, UExact, 14)


    def test_sparse_matrix_patch_test_dirichlet_BC(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1),
                FunctionSpace.EssentialBC(nodeSet='right', component=0)]
        
        dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)
        Ubc = dofManager.get_bc_values(self.U)
        
        def compute_energy_again(Uu, Ubc):
            U = dofManager.create_field(Uu, Ubc)
            return self.compute_energy(U, self.internals)

        
        K = assemble_sparse_stiffness_matrix(self.elementStiffnesses,
                                             self.mesh.conns,
                                             dofManager)
                
        innerLambda = lambda Uu, Ubc, vbc: np.vdot( grad(compute_energy_again,1)(Uu, Ubc), vbc )
        compute_bc_cross_stiffness_action = grad(innerLambda , 0)

        bcVals = np.array([0., 0., 1.0])

        dU = np.zeros(self.U.shape)
        for b, ebc in enumerate(ebcs):
            idx = self.mesh.nodeSets[ebc.nodeSet], ebc.component
            dU = dU.at[idx].set(bcVals[b])

        Ubc = dofManager.get_bc_values(self.U)
        Uu = dofManager.get_unknown_values(self.U)
        
        F = -compute_bc_cross_stiffness_action(Uu, Ubc, dofManager.get_bc_values(dU))
        u = linalg.spsolve(K, F)
        
        dU = dU.at[dofManager.isUnknown].set(u)
        UNew = self.U + dU

        # plane strain, Uy should be -nu/(1-nu)*Ux
        exactSolution = lambda x: np.column_stack((x[:,0], -nu/(1.0 - nu)*x[:,1]))
        UExact = exactSolution(self.mesh.coords)
        self.assertArrayNear(UNew, UExact, 14)

if __name__ == '__main__':
    MeshFixture.unittest.main()
