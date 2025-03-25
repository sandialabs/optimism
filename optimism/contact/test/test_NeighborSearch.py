from optimism.JaxConfig import *
from optimism import Mesh
from optimism.test.MeshFixture import MeshFixture
from optimism.contact import MortarContact
import unittest

class SelfContactPenaltyFixture(MeshFixture):

    def setUp(self):
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])

        m1 = self.create_mesh_and_disp(2, 4, [0.0, 1.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '1')

        # Have second block offest (not penetrating) to test that neighbor search excludes edges on opposite faces of the same body
        # by using maxPenetrationDistance = 1/2 the body length
        m2 = self.create_mesh_and_disp(2, 3, [1.1, 2.1], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '2')
        
        # 6 o ----------- o 7    12 o ----------- o 13
        #   |             |         |             |
        # 4 o ----------- o 5       |             |
        #   |             |      10 o ----------- o 11
        # 2 o ----------- o 3       |             |
        #   |             |         |             |
        # 0 o ----------- o 1     8 o ----------- o 9
        
        self.mesh, _ = Mesh.combine_mesh(m1, m2)
        self.disp = np.zeros(self.mesh.coords.shape)
        self.maxPenetrationDistance = 0.5


    def test_exclusion_of_current_and_adjacent_edges(self):
        coordsSegA = np.array([[0, 1], [1, 3], [3, 5], [5, 7]])
        coordsSegB = np.array([[1, 3]]) 

        numNeighbors = 1
        neighborList = MortarContact.get_closest_neighbors_for_self_contact(coordsSegA, coordsSegB, self.mesh, self.disp, numNeighbors, maxPenetrationDistance=2.0)

        goldNeighbors = np.array([[3]])
        self.assertArrayEqual(neighborList, goldNeighbors)


    def test_exclusion_of_edges_on_opposite_face(self):
        coordsSegA = np.array([[2, 0], [9, 11]])
        coordsSegB = np.array([[1, 3]]) 

        numNeighbors = 1
        neighborList = MortarContact.get_closest_neighbors_for_self_contact(coordsSegA, coordsSegB, self.mesh, self.disp, numNeighbors, self.maxPenetrationDistance)

        goldNeighbors = np.array([[1]])
        self.assertArrayEqual(neighborList, goldNeighbors)
    

    def test_inclusion_of_edges_on_adjacent_face(self):
        coordsSegA = np.array([[0, 1], [1, 3], [3, 5], [12, 10], [10, 8], [5, 7]])
        coordsSegB = np.array([[1, 3]]) 

        numNeighbors = 2
        neighborList = MortarContact.get_closest_neighbors_for_self_contact(coordsSegA, coordsSegB, self.mesh, self.disp, numNeighbors, self.maxPenetrationDistance)

        goldNeighbors = np.array([[4, 3]])
        self.assertArrayEqual(neighborList, goldNeighbors)
    

if __name__ == '__main__':
    unittest.main()
