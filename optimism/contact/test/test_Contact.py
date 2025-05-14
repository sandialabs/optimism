import unittest
from optimism.JaxConfig import *
from optimism.test.MeshFixture import MeshFixture
from optimism.contact import Contact
from optimism.contact import Friction
from optimism import Mesh
from optimism import QuadratureRule

import itertools, operator

def sort_uniq(sequence):
    return np.array(list(map(
        operator.itemgetter(0),
        itertools.groupby(sorted(sequence)))))


frictionParams = Friction.Params(0.3, 1e-4)

class TestContactFrictionData(MeshFixture):

    def setUp(self):
        targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])
        #targetDispGrad = np.zeros((2,2))
        
        m1 = self.create_mesh_and_disp(5, 3, [0., 1.], [0., 1.],
                                       lambda x : targetDispGrad.dot(x),
                                       setNamePostFix='1')
        
        m2 = self.create_mesh_and_disp(6, 2, [0., 1.], [1., 2.],
                                       lambda x : targetDispGrad.dot(x),
                                       setNamePostFix='2')
        
        self.mesh, self.U = Mesh.combine_mesh(m1, m2)
        
        self.surfM = self.mesh.sideSets['top1']
        self.surfI = self.mesh.sideSets['bottom2']
        
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(4)
        
        maxContactNeighbors = 3
        self.interactionList = Contact.get_potential_interaction_list(self.surfM, self.surfI, self.mesh,
                                                                      self.U, maxContactNeighbors)
        self.closestSides,self.sideWeights = \
            Contact.compute_closest_edges_and_field_weights(self.mesh, self.U,
                                                            self.quadRule,
                                                            self.interactionList,
                                                            self.surfI)
        
        
    def test_friction_search_static(self):
        
        coordsQ = Contact.compute_q_coordinates(self.mesh, self.U, self.quadRule, self.surfI)
        coordsFromWeights = Contact.compute_q_coordinates_from_field_weights(self.mesh, self.U,
                                                                             self.closestSides,
                                                                             self.sideWeights)

        self.assertArrayNear(coordsQ, coordsFromWeights, 14)

        self.lam = np.ones_like(coordsQ[:,:,0])
        frictionEnergy = Contact.compute_friction_potential(self.mesh, self.U,
                                                            self.lam, frictionParams,
                                                            self.quadRule, self.surfI,
                                                            self.closestSides, self.sideWeights)

        self.assertNear(frictionEnergy, 0.0, 14)

        
    def test_friction_search_after_motion(self):

        coordsQ0 = Contact.compute_q_coordinates(self.mesh, self.U, self.quadRule, self.surfI)

        block1Nodes = sort_uniq(self.mesh.conns[self.mesh.blocks['block1']].ravel())
        block2Nodes = sort_uniq(self.mesh.conns[self.mesh.blocks['block2']].ravel())
        
        self.U = self.U.at[block1Nodes,0].add(0.1)
        self.U = self.U.at[block2Nodes,0].add(-0.25)
        
        coordsQ = Contact.compute_q_coordinates(self.mesh, self.U, self.quadRule, self.surfI)
        coordsFromWeights = Contact.compute_q_coordinates_from_field_weights(self.mesh, self.U,
                                                                             self.closestSides,
                                                                             self.sideWeights)
        
        self.assertArrayNear(coordsQ0, coordsFromWeights-np.array([0.1, 0.0]), 14)
        self.assertArrayNear(coordsQ0, coordsQ+np.array([0.25, 0.0]), 14)

        
if __name__ == '__main__':
    unittest.main()

    
