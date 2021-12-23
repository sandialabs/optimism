import matplotlib.pyplot as plt
import unittest
import jax.random
import numpy as onp

from optimism.JaxConfig import *
from optimism import Interpolants
from optimism import QuadratureRule
from optimism.test import TestFixture

    
tol = 1e-14


class TestInterpolants(TestFixture.TestFixture):
    
    def setUp(self):
        maxDegree = 5
        self.masters1d = []
        self.mastersTri = []
        for degree in range(1, maxDegree + 1):
            self.masters1d.append(Interpolants.make_master_line_element(degree))
            self.mastersTri.append(Interpolants.make_master_tri_element(degree))


    def test_1D_interpolant_points_in_element(self):
        for master in self.masters1d:
            p = master.coordinates
            self.assertTrue(np.all(p >= 0.0) and np.all(p <= 1.0))


    def test_1D_master_element_topological_nodesets(self):
        for master in self.masters1d:
            p = master.coordinates
            self.assertNear(p[master.vertexNodes[0]], 0.0, 14)
            self.assertNear(p[master.vertexNodes[1]], 1.0, 14)
            
            if master.interiorNodes is not None:
                self.assertTrue(np.all(p[master.interiorNodes] > 0.0))
                self.assertTrue(np.all(p[master.interiorNodes] < 1.0))

            
    def test_tri_interpolant_points_in_element(self):
        for master in self.mastersTri:
            p = master.coordinates
            # x conditions
            self.assertTrue(np.all(p[:,0] >= -tol))
            self.assertTrue(np.all(p[:,0] <= 1.0 + tol))
            # y conditions
            self.assertTrue(np.all(p[:,1] >= -tol))
            self.assertTrue(np.all(p[:,1] <= 1. - p[:,0] + tol))


    def test_tri_master_element_topological_nodesets(self):
        for master in self.mastersTri:
            p = master.coordinates
            self.assertArrayNear(p[master.vertexNodes[0],:],
                                 np.array([1.0, 0.0]),
                                 14)
            self.assertArrayNear(p[master.vertexNodes[1],:],
                                 np.array([0.0, 1.0]),
                                 14)
            self.assertArrayNear(p[master.vertexNodes[2],:],
                                 np.array([0.0, 0.0]),
                                 14)
            
            if master.interiorNodes.size > 0:
                k = master.interiorNodes
                self.assertTrue(np.all(p[k,0] > -tol))
                self.assertTrue(np.all(p[k,1] + p[k,0] - 1. <  tol))


    def test_tri_face_nodes_match_1D_lobatto_nodes(self):
        for master1d, masterTri in zip(self.masters1d, self.mastersTri):
            for faceNodeIds in masterTri.faceNodes:
                # get the triangle face node points directly
                xf = masterTri.coordinates[faceNodeIds,:]
                # affine transformation of 1D node points to triangle face
                p = master1d.coordinates
                x1d = np.outer(1.0 - p, xf[0,:]) + np.outer(p, xf[-1,:])
                # make sure they are the same
                self.assertArrayNear(xf, x1d, 14)


    def test_tri_shape_partition_of_unity(self):
        qr = QuadratureRule.create_quadrature_rule_on_triangle(degree=3)
        nQPts = QuadratureRule.len(qr)
        for master in self.mastersTri:
            shapes = Interpolants.compute_shapes_on_tri(master, qr.xigauss)
            self.assertArrayNear(np.sum(shapes, axis=1), np.ones(nQPts), 14)


    def test_tri_shapeGrads_partition_of_unity(self):
        qr = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        nQPts = QuadratureRule.len(qr)
        for master in self.mastersTri:
            shapeGrads = Interpolants.compute_shapeGrads_on_tri(master, qr.xigauss)
            self.assertArrayNear(np.sum(shapeGrads, axis=1), np.zeros((nQPts,2)), 14)

            
    def test_shape_kronecker_delta_property(self):
        for master in self.mastersTri:
            shapeAtNodes = Interpolants.compute_shapes_on_tri(master, master.coordinates)
            nNodes = master.coordinates.shape[0]
            self.assertArrayNear(shapeAtNodes, np.identity(nNodes), 14)


    def test_interpolation(self):
        x = self.generate_random_points_in_triangle(1)
        for master in self.mastersTri:
            degree = master.degree
            polyCoeffs = np.fliplr(np.triu(np.ones((degree+1,degree+1))))
            expected = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], polyCoeffs)

            shape = Interpolants.compute_shapes_on_tri(master, x)
            fn = onp.polynomial.polynomial.polyval2d(master.coordinates[:,0],
                                                     master.coordinates[:,1],
                                                     polyCoeffs)
            fInterpolated = np.dot(shape, fn)
            self.assertArrayNear(expected, fInterpolated, 14)


    def test_grad_interpolation(self):
        x = self.generate_random_points_in_triangle(1)
        for master in self.mastersTri:
            degree = master.degree
            poly = np.fliplr(np.triu(np.ones((degree+1,degree+1))))

            dShape = Interpolants.compute_shapeGrads_on_tri(master, x)
            fn = onp.polynomial.polynomial.polyval2d(master.coordinates[:,0],
                                                     master.coordinates[:,1],
                                                     poly)
            dfInterpolated = np.einsum('qai,a->qi',dShape, fn)

            # exact x derivative
            direction = 0
            DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
            expected0 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

            self.assertArrayNear(expected0, dfInterpolated[:,0], 14)
            
            direction = 1
            DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
            expected1 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

            self.assertArrayNear(expected1, dfInterpolated[:,1], 14)

            
        
    def no_test_plot_interpolants_tri(self):
        degree = 10
        nodeScheme = Interpolants.make_master_tri_element(degree)
        p = nodeScheme.coordinates
        fig = plt.figure()
        plt.scatter(p[:,0], p[:,1])
        plt.gca().axis('equal')
        plt.show()


    def generate_random_points_in_triangle(self, npts):
        key = jax.random.PRNGKey(2)
        x = jax.random.uniform(key, (npts,))
        y = np.zeros(npts)
        for i in range(npts):
            key,subkey = jax.random.split(key)
            y = ops.index_update(y,
                                 ops.index[i],
                                 jax.random.uniform(subkey, minval=0.0, maxval=1.0-x[i]))
                     
        return np.column_stack((x,y))


class TestBubbleInterpolants(TestInterpolants):
    def setUp(self):
        maxDegree = 3
        self.masters1d = []
        self.mastersTri = []
        for degree in range(2, maxDegree + 1):
            self.masters1d.append(Interpolants.make_master_line_element(degree))
            self.mastersTri.append(Interpolants.make_master_tri_bubble_element(degree))


    def no_test_plot_shape_functions(self):
        s = self.generate_random_points_in_triangle(500)
        shape = Interpolants.compute_shapes_on_bubble_tri(self.mastersTri[1], s)
        unit = np.sum(shape, axis=1)
        print(np.max(unit), np.min(unit))

        for i in range(Interpolants.num_nodes(self.mastersTri[1])):
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(s[:,0], s[:,1], shape[:,i])

        plt.show()


if __name__ == '__main__':
    unittest.main()
