import unittest
from optimism.JaxConfig import *
from optimism import Mesh
from optimism import QuadratureRule
from optimism import Surface
from optimism.contact import EdgeIntersection
from optimism import FunctionSpace
from optimism.test.MeshFixture import MeshFixture
from optimism.contact import Contact
from matplotlib import pyplot as plt


def get_side_set_segments(mesh, sideset, disp):

    def get_current_edge_segments(edge):
        XOnEdge = Mesh.get_edge_coords(mesh, edge)
        uOnEdge = Mesh.get_edge_field(mesh, edge, disp)
        xOnEdge = XOnEdge + uOnEdge
        return vmap(lambda x, y: np.array([x,y]))(xOnEdge[:-1], xOnEdge[1:])

    segments = vmap(get_current_edge_segments)(sideset)
    return segments.reshape((segments.shape[0]*segments.shape[1], segments.shape[2], segments.shape[3]))


def get_best_overlap_vector(mesh, edge, listOfEdges):
    edgeCoords = Surface.get_coords(mesh, edge)
    normal = Surface.compute_normal(edgeCoords)
    edgeCenter = np.average(edgeCoords, axis=0)
    
    integrationRay = np.array([edgeCenter, normal])
    
    minDistance = np.inf
    for neighbor in listOfEdges:
        if not np.all(edge==neighbor):
            neighborCoords = Surface.get_coords(mesh, neighbor)
            distance, parCoord = EdgeIntersection.compute_valid_ray_trace_distance(neighborCoords, integrationRay)
            minDistance = distance if distance < minDistance else minDistance

    return minDistance * integrationRay[1]


class TwoBodyContactFixture(MeshFixture):

    def setUp(self):
        self.tol = 1e-7
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])
        
        m1 = self.create_mesh_and_disp(3, 5, [0.0, 1.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '1')

        m2 = self.create_mesh_and_disp(2, 4, [0.9, 2.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '2')
        
        self.mesh, _ = Mesh.combine_mesh(m1, m2)
        order=2
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=order, copyNodeSets=False)
        self.disp = np.zeros(self.mesh.coords.shape)


        edgesTop1 = get_side_set_segments(self.mesh, self.mesh.sideSets['right1'], self.disp)
        edgesTop2 = get_side_set_segments(self.mesh, self.mesh.sideSets['left2'], self.disp)

        plt.clf()
        plt.plot( edgesTop1[:,:,0].T, edgesTop1[:,:,1].T, linestyle='-', color='y', markerfacecolor='red', marker='o')
        plt.plot( edgesTop2[:,:,0].T, edgesTop2[:,:,1].T, linestyle='-', color='g', markerfacecolor='blue', marker='o')
        plt.show()
        exit(1)

        nodeSets = Mesh.create_nodesets_from_sidesets(self.mesh)
        self.mesh = Mesh.mesh_with_nodesets(self.mesh, nodeSets)
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(3)
        

    @unittest.skipIf(True, '')
    def test_combining_nodesets(self):
        self.assertArrayEqual( self.mesh.nodeSets['top1'], [12,13,14] )
        numNodesMesh1 = 15
        self.assertArrayEqual( self.mesh.nodeSets['top2'], np.array([6,7])+numNodesMesh1 )

    @unittest.skipIf(False, '')
    def test_combining_sidesets(self):
        self.assertArrayEqual( self.mesh.sideSets['top1'], np.array([[7,1],[15,1]]) )
        numElemsMesh1 = 2*8
        self.assertArrayEqual( self.mesh.sideSets['top2'], np.array([[5+numElemsMesh1,1]]) )

        
    def plot_solution(self, plotName):
        import VTKWriter

        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=self.disp,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        
        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs',
                               nodalData=bcs,
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)
        writer.write()
        

    @unittest.skipIf(True, '')
    def test_contact_search(self):
        self.disp = 0.0*self.disp
        
        sideA = self.mesh.sideSets['right1']
        sideB = self.mesh.sideSets['left2']
        
        interactionList = Contact.get_potential_interaction_list(sideA, sideB, self.mesh, self.disp, 3)
        
        minDists = Contact.compute_closest_distance_to_each_side(self.mesh, self.disp, self.quadRule,
                                                                 interactionList,
                                                                 sideB)
        
        self.assertArrayNear(minDists, -0.1*np.ones((3,2)), 9)


if __name__ == '__main__':
    unittest.main()

