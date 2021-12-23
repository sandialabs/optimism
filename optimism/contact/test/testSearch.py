import unittest
from optimism.JaxConfig import *
from optimism import Mesh
from optimism import Surface
from optimism import QuadratureRule
from optimism.contact import Search
from optimism.contact import EdgeIntersection
from optimism.test.MeshFixture import MeshFixture


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


class TestDoubleMeshFixture(MeshFixture):

    def setUp(self):
        self.tol = 1e-7
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])
        
        m1 = self.create_mesh_and_disp(3, 5, [0., 1.], [0., 1.],
                                       lambda x : self.targetDispGrad.dot(x))

        m2 = self.create_mesh_and_disp(2, 4, [1.1, 2.], [0.,1.],
                                       lambda x : self.targetDispGrad.dot(x))

        self.mesh, self.disp = Mesh.combine_mesh(m1,m2)

                    
    def is_integration_edge(self, x):
        tol = self.tol
        return np.all( x[:,0] > 1.0 - tol ) and np.all( x[:,0] < 1.0 + tol )

    
    def is_contact_edge(self, x):
        tol = self.tol
        return np.all( x[:,0] > 1.0 - tol ) and np.all( x[:,0] < 1.1 + tol )

 
    def test_correct_number_of_edges_created_for_contact(self):
        edges = Surface.create_edges(self.mesh.coords, self.mesh.conns, self.is_contact_edge)

        for edge in edges:
            coords = Surface.get_coords(self.mesh, edge)
        
        self.assertEqual(7, edges.shape[0])
        

    def test_surface_integral_of_linears(self):
        edges = Surface.create_edges(self.mesh.coords, self.mesh.conns, self.is_integration_edge)
        quadratureRule = QuadratureRule.create_quadrature_rule_1D(1)
        I = 0.0
        for edge in edges:
            coords = Surface.get_coords(self.mesh, edge)
            I += Surface.integrate_function(quadratureRule, coords, lambda x: x[1])
        self.assertNear(I, 0.5, 14)

        
    def test_surface_integral_of_quadratics(self):
        edges = Surface.create_edges(self.mesh.coords, self.mesh.conns, self.is_integration_edge)
        quadratureRule = QuadratureRule.create_quadrature_rule_1D(2)
        I = 0.0
        for edge in edges:
            coords = Surface.get_coords(self.mesh, edge)
            I += Surface.integrate_function(quadratureRule, coords, lambda x: 3.*x[1]**2)
        self.assertNear(I, 1.0, 14)

        
    def test_contact_distance_constraint_evaluation(self):
        mesh = self.mesh
        edges = Surface.create_edges(self.mesh.coords, self.mesh.conns, self.is_contact_edge)
        coords = mesh.coords
        
        for edge in edges:
            minOverlapVector = get_best_overlap_vector(mesh, edge, edges)

            edgeCoords = Surface.get_coords(mesh, edge)
            if edgeCoords[0][0] < 1 + self.tol:
                self.assertArrayNear(minOverlapVector, np.array([0.1, 0.0]), 12)
            elif edgeCoords[0][0] > 1.1 - self.tol:
                self.assertArrayNear(minOverlapVector, np.array([-0.1, 0.0]), 12)
            else:
                print("There should be no other edges\n")
                
                    
if __name__ == '__main__':
    unittest.main()
