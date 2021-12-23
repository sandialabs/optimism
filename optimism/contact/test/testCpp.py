import unittest
import matplotlib.pyplot as plt

from optimism.JaxConfig import *

from optimism.contact import EdgeCpp
from optimism.contact import SmoothMinMax
from optimism.test.TestFixture import TestFixture

doPlotting=False

def compute_grid_field(N, xmin, xmax, evaluation_function):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(xmin, xmax, N)
    
    X,Y = np.meshgrid(x,y)
    V = np.array([vmap(lambda xi: np.array([xi,yi]))(x) for yi in y])
    Z = vmap(lambda vs: vmap(lambda v: evaluation_function(v))(vs))(V)
    
    return X,Y,Z


def edges_from_points(p0, p1, p2):
    edge1 = np.array([p0, p1])
    edge2 = np.array([p1, p2])
    return np.array([edge1, edge2])


class TestEdgeIntersection(TestFixture):

    def setUp(self):

        #self.v1 = np.array([0., 0.])
        #self.v2 = np.array([ 1., 1.])
        #self.v3 = np.array([ 1., 0,])

        self.v1 = np.array([-1., -1.])
        self.v2 = np.array([ 1., 1.])
        self.v3 = np.array([ 1., -1,])

        self.edges = edges_from_points(self.v3, self.v2, self.v1)
        self.edgesR = edges_from_points(self.v1, self.v2, self.v3)

        self.xmin = -1.5
        self.xmax = 1.5
        
        
    def test_cpp_dist_interior(self):
        p = np.array([0.5, -0.5])
        dist = EdgeCpp.cpp_distance(self.edges[1], p)
        self.assertNear(dist, -np.sqrt(0.5), 14)


    def test_cpp_dist_exterior(self):
        p = np.array([-0.5, 0.5])
        dist = EdgeCpp.cpp_distance(self.edges[1], p)
        self.assertNear(dist, np.sqrt(0.5), 14)

        
    def test_cpp_dist_corner1(self):
        p = np.array([1.5, 1.5])
        dist = EdgeCpp.cpp_distance(self.edges[1], p)
        self.assertNear(dist, np.sqrt(0.5), 14)

        
    def test_cpp_dist_corner2(self):
        p = np.array([2.0, 1.0])
        dist = EdgeCpp.cpp_distance(self.edges[1], p)
        self.assertNear(dist, -1.0, 14)


    def plot_grid(self, X, Y, Z, edges, outname):

        if doPlotting:
            plt.clf()
            fig1, ax2 = plt.subplots(constrained_layout=True)
            c = ax2.contourf(X, Y, Z, levels=[-0.6+0.1*i for i in range(13)])

            ax2.set_title('Distance contours')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')

            cbar = fig1.colorbar(c)
            cbar.ax.set_ylabel('distance')

            edge0 = edges[0]
            edge1 = edges[1]
        
            plt.plot(edge0[:,0],
                     edge0[:,1],
                     color='k', linestyle='-', linewidth=2)
        
            plt.plot(edge1[:,0],
                     edge1[:,1],
                     color='k', linestyle='-', linewidth=2)
        
            plt.axis([self.xmin, self.xmax, self.xmin, self.xmax])

            ax2.set_aspect('equal', adjustable='box')
            if outname is not None:
                plt.savefig(outname)

            
    def test_smooth_1(self):
        N = 301
        tol = 1e-1
        
        X,Y,Z = compute_grid_field(N, self.xmin, self.xmax, lambda p: EdgeCpp.smooth_distance(self.edges, p, tol))
        self.plot_grid(X, Y, Z, self.edges, 'smooth1.png')


    def test_smooth_2(self):
        N = 301
        tol = 1e-1
        
        X,Y,Z = compute_grid_field(N, self.xmin, self.xmax, lambda p: EdgeCpp.smooth_distance(self.edgesR, p, tol))
        self.plot_grid(X, Y, Z, self.edgesR, 'smooth2.png')


    def test_limits(self):
        N = 201
        tol = 6e-1
        Ys = np.linspace(1.8, 2.2, 21)

        v1 = np.array([0., 0.])
        v2 = np.array([ 1., 1.])
        
        for i,y in enumerate(Ys):
            v3 = np.array([2.0, y])
            edges = edges_from_points(v3, v2, v1)
            smooth_dist = lambda p: EdgeCpp.smooth_distance(edges, p, tol)
            dist = lambda p: grad(EdgeCpp.smooth_distance,0)(edges, p, tol)[0,0,0]
            #dist = smooth_dist
            X,Y,Z = compute_grid_field(N, self.xmin, self.xmax,
                                       dist)
            self.plot_grid(X, Y, Z, edges, 'smooth_'+str(i).zfill(2)+'.png')
            

    def test_plot_smooth_min(self):
        N = 201
        x = np.linspace(0.,1.,N)
        y = 0.5

        func = grad(SmoothMinMax.safe_min,2)
        
        for eps in np.linspace(1e-5, 1e-1, 3):
            m = vmap(func, (0,None,None))(x,y,eps)
            plt.plot(x,m)

        m = vmap(func, (0,None,None))(x,y,0.)
        plt.plot(x,m, '--k')

        if doPlotting:
            plt.savefig('smooth.png')
            
            
if __name__ == '__main__':
    unittest.main()
