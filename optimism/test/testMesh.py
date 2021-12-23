from optimism.JaxConfig import *
from optimism.test import MeshFixture
from optimism import Mesh
from optimism import Interpolants
#from optimism.Mesh import EssentialBC
#from optimism.Mesh import DofManager

#from matplotlib import pyplot as plt


class TestSingleMeshFixture(MeshFixture.MeshFixture):
    
    def is_constrained_in_xy(self, x, xRange, yRange):
        tol = 1e-7
        onBoundary = x[0] < xRange[0] + tol or x[0] > xRange[1] - tol or x[1] < yRange[0] + tol or x[1] > yRange[1] - tol
        return [onBoundary, onBoundary]

    
    def setUp(self):
        self.Nx = 4
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]

        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        
        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x : self.targetDispGrad.dot(x))

        #EBCs = []
        #EBCs.append(EssentialBC(nodeSet='all_boundary', field=0))
        #EBCs.append(EssentialBC(nodeSet='all_boundary', field=1))
        #self.dofManager = DofManager(self.mesh, self.U.shape, EBCs)                               

    def test_create_nodesets_from_sidesets(self):
        master = Interpolants.make_master_tri_element(1)
        nodeSets = Mesh.create_nodesets_from_sidesets(self.mesh)

        # this test relies on the fact that matching nodesets
        # and sidesets were created on the MeshFixture
        
        for key in self.mesh.sideSets:
            self.assertArrayEqual(np.sort(self.mesh.nodeSets[key]), nodeSets[key])

            
    #
    # TODO: test DofManager operations          
    #

    
    def disable_test_edge_conn(self):
        print('coords=\n',self.mesh.coords)
        print('conns=\n', self.mesh.conns)
        edgeConns,edges = Mesh.create_edges(self.mesh.coords, self.mesh.conns)
        print('edgeConns=\n',edgeConns)
        print('edges=\n', edges)


    def disable_test_higher_order_mesh_creation(self):
        # case with no interior nodes
        master, master1d = Interpolants.make_master_elements(2)
        newMesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, master, master1d)

        print('coords=\n', newMesh.coords)
        print('conns=\n', newMesh.conns)
        plt.triplot(self.mesh.coords[:,0], self.mesh.coords[:,1], newMesh.conns[:,master.vertexNodes])
        plt.scatter(newMesh.coords[:,0], newMesh.coords[:,1], marker='o')

        # case with interior nodes
        master, master1d = Interpolants.make_master_elements(3)
        newMesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, master, master1d)

        print('coords=\n', newMesh.coords)
        print('conns=\n', newMesh.conns)
        plt.figure()
        plt.triplot(self.mesh.coords[:,0], self.mesh.coords[:,1], newMesh.conns[:,master.vertexNodes])
        plt.scatter(newMesh.coords[:,0], newMesh.coords[:,1], marker='o')
        plt.show()


            
if __name__ == '__main__':
    MeshFixture.unittest.main()

