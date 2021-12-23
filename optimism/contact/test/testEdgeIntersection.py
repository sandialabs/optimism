import matplotlib.pyplot as plt
import unittest

from optimism.JaxConfig import *
from optimism.contact import EdgeIntersection
from optimism.test.TestFixture import TestFixture

class TestEdgeIntersection(TestFixture):

    def setUp(self):
        self.v1 = np.array([-1.,-1.])
        self.v2 = np.array([1.,1.])
        self.p = np.array([1.,-1.])
        self.raySmoothing = 1e-5

        
    def get_edge(self):
        return np.array([self.v1, self.v2])

    
    def get_ray(self, normal):
        return np.array([self.p, normal]) 


    def compute_ray_trace(self, normal):
        edge = self.get_edge()
        ray = self.get_ray(normal)
        return EdgeIntersection.compute_valid_ray_trace_distance_smoothed(edge, ray, self.raySmoothing)

        
    def test_valid_intersection(self):
        normal = np.array([-1.,1.])
        
        rayLength,edgeParCoord = self.compute_ray_trace(normal)
                                                        
        pointOnEdge = self.v1 + edgeParCoord*(self.v2-self.v1)
        pointOnRay = self.p + rayLength*normal

        self.assertArrayEqual(pointOnEdge, pointOnRay)
        self.assertArrayEqual(pointOnEdge, np.array([0.,0.]))
        self.assertEqual(edgeParCoord, 0.5)
        self.assertEqual(rayLength , 1.0)


    def test_valid_intersection_on_edge(self):
        normal = np.array([0,1.])

        rayLength,edgeParCoord = self.compute_ray_trace(normal)
        pointOnEdge = self.v1 + edgeParCoord*(self.v2-self.v1)
        pointOnRay = self.p + rayLength*normal

        self.assertArrayEqual(pointOnEdge, pointOnRay)
        self.assertArrayEqual(pointOnEdge, np.array([1.,1.]))
        self.assertEqual(edgeParCoord, 1.0)
        self.assertEqual(rayLength, 2.0)

        
    def test_smooth_gradient_on_either_side_of_right_edge(self):
        ray_trace = lambda normal: self.compute_ray_trace(normal)[0]
        ray_trace_grad = grad(ray_trace)

        eps = 1e-15
        normalIn = np.array([-eps,1.])
        normalOut = np.array([+eps,1.])

        self.assertNear(ray_trace(normalIn), ray_trace(normalOut), 13)
        self.assertArrayNear(ray_trace_grad(normalIn), ray_trace_grad(normalOut), 2)

        
    def test_smooth_gradient_on_either_side_of_left_edge(self):
        ray_trace = lambda normal: self.compute_ray_trace(normal)[0]
        ray_trace_grad = grad(ray_trace)

        eps = 1e-15
        normalIn = np.array([-1.,eps])
        normalOut = np.array([-1.,-eps])
        
        self.assertNear(ray_trace(normalIn), ray_trace(normalOut), 13)
        self.assertArrayNear(ray_trace_grad(normalIn), ray_trace_grad(normalOut), 2)
        

    def test_limit_of_ray_smoothing(self):
        ray_trace = lambda normal: self.compute_ray_trace(normal)[0]

        normalRight = np.array([0.99*self.raySmoothing, 1.])
        self.assertTrue(ray_trace(normalRight) > 1.0)
        self.assertTrue(ray_trace(normalRight) != np.inf)

        normalMoreRight = np.array([0.99999*self.raySmoothing, 1.])
        self.assertTrue(ray_trace(normalMoreRight) > 1.0)
        self.assertTrue(ray_trace(normalMoreRight) != np.inf)
        
        normalWayRight = np.array([1.01*self.raySmoothing, 1.])
        self.assertTrue(ray_trace(normalWayRight) > 1.0)

        normalLeft = np.array([-1., -0.99*self.raySmoothing])
        self.assertTrue(ray_trace(normalLeft) > 1.0)
        self.assertTrue(ray_trace(normalLeft) != np.inf)

        normalMoreLeft = np.array([-1., -0.99999*self.raySmoothing])
        self.assertTrue(ray_trace(normalMoreLeft) > 1.0)
        self.assertTrue(ray_trace(normalMoreLeft) != np.inf)
        
        normalWayLeft = np.array([-1., -1.01*self.raySmoothing])
        self.assertTrue(ray_trace(normalWayLeft) > 1.0)
        
        
    def get_ray_length_arg_x(self, xComp):
        sol = EdgeIntersection.compute_intersection(self.get_edge(), self.get_ray([xComp, 1.0]))
        return sol[1]

    
    def get_ray_length_arg_y(self, yComp):
        sol = EdgeIntersection.compute_intersection(self.get_edge(), self.get_ray([-1.0, yComp]))
        return sol[1]

    
    @unittest.skip
    def test_plot(self):
        N = 150
        xcomp = np.linspace(-0.5, 0.5, N)
        rays = [self.get_ray_length_arg_x(x) for x in xcomp]
        plt.plot(xcomp, rays)
        plt.show()

        
    @unittest.skip
    def test_plot2(self):
        N = 150
        ycomp = np.linspace(-0.5, 0.5, N)
        rays = [self.get_ray_length_arg_y(x) for x in ycomp]
        plt.plot(ycomp, rays)
        plt.show()

        rayDer = [jit(grad(self.get_ray_length_arg_y))(x) for x in ycomp]
        plt.plot(ycomp, rayDer)
        plt.show()

        
if __name__ == '__main__':
    unittest.main()


