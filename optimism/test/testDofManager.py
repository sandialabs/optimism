from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import QuadratureRule
from optimism.test import MeshFixture



class DofManagerTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 4
        self.Ny = 5
        xRange = [0.,1.]
        yRange = [0.,1.]
        
        #self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                 lambda x : 0*x)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(1)
        fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        ebcs = [FunctionSpace.EssentialBC(nodeSet='top', component=0),
                FunctionSpace.EssentialBC(nodeSet='right', component=1)]

        self.nNodes = self.Nx*self.Ny
        self.nFields = 2
        self.dofManager = FunctionSpace.DofManager(fs, self.nFields, ebcs)

        self.nDof = self.nFields*self.nNodes
        U = np.zeros((self.nNodes, self.nFields))
        U = U.at[:,1].set(1.0)
        U = U.at[self.mesh.nodeSets['top'],0].set(2.0)
        self.U = U.at[self.mesh.nodeSets['right'],1].set(3.0)
        
    def test_get_bc_size(self):
        # number of dofs from top, field 0
        nEbcs = self.Nx
        # number of dofs from right, field 1
        nEbcs += self.Ny
        self.assertEqual(self.dofManager.get_bc_size(), nEbcs)


    def test_get_unknown_size(self):
        # number of dofs from top, field 0
        nEbcs = self.Nx
        # number of dofs from right, field 1
        nEbcs += self.Ny
        self.assertEqual(self.dofManager.get_unknown_size(), self.nDof - nEbcs)


    def test_slice_unknowns_with_dof_indices(self):
        Uu = self.dofManager.get_unknown_values(self.U)
        Uu_x = self.dofManager.slice_unknowns_with_dof_indices(Uu, (slice(None),0) )
        self.assertArrayEqual(Uu_x, np.zeros(self.Nx*(self.Ny-1)))
        Uu_y = self.dofManager.slice_unknowns_with_dof_indices(Uu, (slice(None),1) )
        self.assertArrayEqual(Uu_y, np.ones(self.Ny*(self.Nx-1)))
        
        
if __name__ == '__main__':
    MeshFixture.unittest.main()
