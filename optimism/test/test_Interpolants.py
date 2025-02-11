import matplotlib.pyplot as plt
import numpy as onp
import unittest

import jax
import jax.numpy as np

from optimism import Interpolants
from optimism import QuadratureRule
import TestFixture

    
tol = 1e-14

def generate_random_points_in_triangle(npts):
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (npts,))
    y = jax.numpy.zeros(npts)
    for i in range(npts):
        key,subkey = jax.random.split(key)
        y = y.at[i].set(jax.random.uniform(subkey, minval=0.0, maxval=1.0-x[i]))
    points = jax.numpy.column_stack((x,y))
    return  onp.asarray(points)

class ElementFixture(TestFixture.TestFixture):
    def assertCorrectInterpolationOn2DElement(self, element, numRandomEvalPoints, shape2dFun):
        degree = element.degree
        x = generate_random_points_in_triangle(numRandomEvalPoints)
        polyCoeffs = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))

        shape, _ = shape2dFun(degree, element.coordinates, x)
        fn = onp.polynomial.polynomial.polyval2d(element.coordinates[:,0],
                                                 element.coordinates[:,1],
                                                 polyCoeffs)
        fInterpolated = onp.dot(shape, fn)

        expected = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], polyCoeffs)
        self.assertArrayNear(expected, fInterpolated, 14)

    def assertCorrectGradInterpolationOn2DElement(self, element, numRandomEvalPoints, shape2dFun):
        degree = element.degree
        x = generate_random_points_in_triangle(numRandomEvalPoints)
        poly = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))

        _, dShape = shape2dFun(degree, element.coordinates, x)
        fn = onp.polynomial.polynomial.polyval2d(element.coordinates[:,0],
                                                 element.coordinates[:,1],
                                                 poly)
        dfInterpolated = onp.einsum('qai,a->qi',dShape, fn)

        # exact x derivative
        direction = 0
        DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
        expected0 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

        self.assertArrayNear(expected0, dfInterpolated[:,0], 14)

        direction = 1
        DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
        expected1 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

        self.assertArrayNear(expected1, dfInterpolated[:,1], 14)


class TestInterpolants(ElementFixture):
    
    def setUp(self):
        maxDegree = 5
        self.elements1d = []
        self.elements = []
        self.qr1d = QuadratureRule.create_quadrature_rule_1D(degree=maxDegree)
        self.nQPts1d = len(self.qr1d)
        self.qr = QuadratureRule.create_quadrature_rule_on_triangle(degree=maxDegree)
        self.nQPts = len(self.qr)
        for degree in range(1, maxDegree + 1):
            self.elements1d.append(Interpolants.make_parent_element_1d(degree))
            self.elements.append(Interpolants.make_parent_element_2d(degree))


    def test_1D_interpolant_points_in_element(self):
        for element in self.elements1d:
            p = element.coordinates
            self.assertTrue(np.all(p >= 0.0) and onp.all(p <= 1.0))


    def test_1D_element_element_topological_nodesets(self):
        for element in self.elements1d:
            p = element.coordinates
            self.assertNear(p[element.vertexNodes[0]], 0.0, 14)
            self.assertNear(p[element.vertexNodes[1]], 1.0, 14)
            
            if element.interiorNodes is not None:
                self.assertTrue(np.all(p[element.interiorNodes] > 0.0))
                self.assertTrue(np.all(p[element.interiorNodes] < 1.0))

            
    def test_tri_interpolant_points_in_element(self):
        for element in self.elements:
            p = element.coordinates
            # x conditions
            self.assertTrue(np.all(p[:,0] >= -tol))
            self.assertTrue(np.all(p[:,0] <= 1.0 + tol))
            # y conditions
            self.assertTrue(np.all(p[:,1] >= -tol))
            self.assertTrue(np.all(p[:,1] <= 1. - p[:,0] + tol))


    def test_tri_element_element_topological_nodesets(self):
        for element in self.elements:
            p = element.coordinates
            self.assertArrayNear(p[element.vertexNodes[0],:],
                                 onp.array([1.0, 0.0]),
                                 14)
            self.assertArrayNear(p[element.vertexNodes[1],:],
                                 onp.array([0.0, 1.0]),
                                 14)
            self.assertArrayNear(p[element.vertexNodes[2],:],
                                 onp.array([0.0, 0.0]),
                                 14)
            
            if element.interiorNodes.size > 0:
                k = element.interiorNodes
                self.assertTrue(np.all(p[k,0] > -tol))
                self.assertTrue(np.all(p[k,1] + p[k,0] - 1. <  tol))


    def test_tri_face_nodes_match_1D_lobatto_nodes(self):
        for element1d, elementTri in zip(self.elements1d, self.elements):
            for faceNodeIds in elementTri.faceNodes:
                # get the triangle face node points directly
                xf = elementTri.coordinates[faceNodeIds,:]
                # affine transformation of 1D node points to triangle face
                p = element1d.coordinates
                x1d = np.outer(1.0 - p, xf[0,:]) + np.outer(p, xf[-1,:])
                # make sure they are the same
                self.assertArrayNear(xf, x1d, 14)


    def test_tri_shape_partition_of_unity(self):
        for element in self.elements:
            shapes, _ = Interpolants.shape2d(element.degree, element.coordinates, self.qr.xigauss)
            self.assertArrayNear(np.sum(shapes, axis=1), np.ones(self.nQPts), 14)


    def test_tri_shapeGrads_partition_of_unity(self):
        for element in self.elements:
            _, shapeGradients = Interpolants.shape2d(element.degree, element.coordinates, self.qr.xigauss)
            self.assertArrayNear(np.sum(shapeGradients, axis=1), np.zeros((self.nQPts, 2)), 14)

            
    def test_shape_kronecker_delta_property(self):
        for element in self.elements:
            shapeAtNodes, _ = Interpolants.shape2d(element.degree, element.coordinates, element.coordinates)
            nNodes = element.coordinates.shape[0]
            self.assertArrayNear(shapeAtNodes, np.identity(nNodes), 14)


    def test_interpolation(self):
        numRandomEvalPoints = 1
        for element in self.elements:
            self.assertCorrectInterpolationOn2DElement(element, numRandomEvalPoints, Interpolants.shape2d)


    def test_grad_interpolation(self):
        numRandomEvalPoints = 1
        for element in self.elements:
            self.assertCorrectGradInterpolationOn2DElement(element, numRandomEvalPoints, Interpolants.shape2d)


    def no_test_plot_high_order_nodes(self):
        degree = 10
        nodeScheme = Interpolants.make_parent_element_2d(degree)
        p = nodeScheme.coordinates
        fig = plt.figure()
        plt.scatter(p[:,0], p[:,1])
        plt.gca().axis('equal')
        plt.show()


class TestBubbleInterpolants(TestFixture.TestFixture):
    def setUp(self):
        maxDegree = 5
        self.elements = []
        self.qr = QuadratureRule.create_quadrature_rule_on_triangle(degree=maxDegree)
        self.nQPts = len(self.qr)
        for degree in range(2, maxDegree + 1):
            self.elements.append(Interpolants.make_parent_element_2d_with_bubble(degree))

    def test_bubble_interpolation(self):
        x = generate_random_points_in_triangle(1)
        for element in self.elements:
            with self.subTest(i=element.degree):
                degree = element.degree
                polyCoeffs = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))
                expected = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], polyCoeffs)

                shape, _ = Interpolants.shape2dBubble(element, x)
                fn = onp.polynomial.polynomial.polyval2d(element.coordinates[:,0],
                                                         element.coordinates[:,1],
                                                         polyCoeffs)
                fInterpolated = onp.dot(shape, fn)
                self.assertArrayNear(expected, fInterpolated, 14)

    def test_bubble_grad_interpolation(self):
        x = generate_random_points_in_triangle(1)
        for element in self.elements:
            degree = element.degree
            poly = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))

            _, dShape = Interpolants.shape2dBubble(element, x)
            fn = onp.polynomial.polynomial.polyval2d(element.coordinates[:,0],
                                                     element.coordinates[:,1],
                                                     poly)
            dfInterpolated = onp.einsum('qai,a->qi',dShape, fn)

            # exact x derivative
            direction = 0
            DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
            expected0 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

            self.assertArrayNear(expected0, dfInterpolated[:,0], 13)

            direction = 1
            DPoly = onp.polynomial.polynomial.polyder(poly, 1, scl=1, axis=direction)
            expected1 = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], DPoly)

            self.assertArrayNear(expected1, dfInterpolated[:,1], 13)

    def no_test_plot_shape_functions(self):
        i = 1
        print('degree=', self.elements[i].degree)

        x = generate_random_points_in_triangle(500)
        shape, _ = Interpolants.shape2dBubble(self.elements[i], x)

        for i in range(Interpolants.num_nodes(self.elements[i])):
            ax = plt.figure().add_subplot(projection='3d')
            ax.stem(x[:,0], x[:,1], shape[:,i])

        plt.show()

class TestLagrangeInterpolants(ElementFixture):

    def setUp(self):
        self.degree = 2 # only second degree is implemented
        self.qr1d = QuadratureRule.create_quadrature_rule_1D(degree=2)
        self.nQPts1d = len(self.qr1d)
        self.qr = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.nQPts = len(self.qr)
    
    def test_degree_other_than_two_throws(self):
        badDegrees = [1, 3, 4, 5]
        dummyCoords = np.array([[0.0, 1.0], [1.0, 0.0]])
        for degree in badDegrees:
            self.assertRaises(NotImplementedError, Interpolants.make_lagrange_parent_element_1d, degree)
            self.assertRaises(NotImplementedError, Interpolants.make_lagrange_parent_element_2d, degree)
            self.assertRaises(NotImplementedError, Interpolants.shape1d_lagrange, degree, dummyCoords, self.qr1d)
            self.assertRaises(NotImplementedError, Interpolants.shape2d_lagrange, degree, dummyCoords, self.qr)

    def test_1D_interpolant_points(self):
        element = Interpolants.make_lagrange_parent_element_1d(self.degree)
        p = element.coordinates
        self.assertTrue(np.all(p >= 0.0) and onp.all(p <= 1.0))
        self.assertNear(p[element.interiorNodes], 0.5, 16)
        self.assertArrayNear(p[element.vertexNodes], onp.array([0.0, 1.0]), 16)
        self.assertTrue(element.faceNodes is None)

    def test_tri_interpolant_points(self):
        element = Interpolants.make_lagrange_parent_element_2d(self.degree)
        p = element.coordinates
        # x conditions
        self.assertTrue(np.all(p[:,0] >= -tol))
        self.assertTrue(np.all(p[:,0] <= 1.0 + tol))
        # y conditions
        self.assertTrue(np.all(p[:,1] >= -tol))
        self.assertTrue(np.all(p[:,1] <= 1. - p[:,0] + tol))
        # vertex node conditions
        self.assertArrayNear(p[element.vertexNodes[0],:], onp.array([1.0, 0.0]), 16)
        self.assertArrayNear(p[element.vertexNodes[1],:], onp.array([0.0, 1.0]), 16)
        self.assertArrayNear(p[element.vertexNodes[2],:], onp.array([0.0, 0.0]), 16)
        # face node conditions
        k = element.faceNodes
        self.assertTrue(np.all(p[k,0] > -tol))
        self.assertTrue(np.all(p[k,1] + p[k,0] - 1. <  tol))
        self.assertArrayNear(p[k[0,:],:], onp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]), 16)
        self.assertArrayNear(p[k[1,:],:], onp.array([[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]]), 16)
        self.assertArrayNear(p[k[2,:],:], onp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]), 16)
        # interior node conditions
        self.assertTrue(element.interiorNodes.shape == (0,))

    def test_edge_shape_kronecker_delta_property(self):
        element = Interpolants.make_lagrange_parent_element_1d(self.degree)
        shapeAtNodes, _ = Interpolants.shape1d_lagrange(element.degree, element.coordinates, element.coordinates)
        nNodes = element.coordinates.shape[0]
        self.assertArrayNear(shapeAtNodes, np.identity(nNodes), 14)

    def test_edge_shape_partition_of_unity(self):
        element = Interpolants.make_lagrange_parent_element_1d(self.degree)
        shapes, _ = Interpolants.shape1d_lagrange(element.degree, element.coordinates, self.qr1d.xigauss)
        self.assertArrayNear(np.sum(shapes, axis=0), np.ones(self.nQPts1d), 14)

    def test_edge_shapeGrads_partition_of_unity(self):
        element = Interpolants.make_lagrange_parent_element_1d(self.degree)
        _, shapeGradients = Interpolants.shape1d_lagrange(element.degree, element.coordinates, self.qr1d.xigauss)
        self.assertArrayNear(np.sum(shapeGradients, axis=0), np.zeros(self.nQPts1d), 14)

    def test_edge_interpolation(self):
        numRandomEvalPoints = 5
        x = onp.random.uniform(low=0.0, high=1.0, size=(numRandomEvalPoints)) # random points on unit interval
        coeffs = onp.random.uniform(low=0.0, high=1.0, size=(3)) # random polynomial coefficients on unit interval

        element = Interpolants.make_lagrange_parent_element_1d(self.degree)
        p = element.coordinates
        shape, _ = Interpolants.shape1d_lagrange(element.degree, p, x)
        fn = onp.polynomial.polynomial.polyval(p, coeffs)
        fInterpolated = onp.dot(fn, shape)

        expected = onp.polynomial.polynomial.polyval(x, coeffs)
        self.assertArrayNear(expected, np.array(fInterpolated), 14)

    def test_edge_grad_interpolation(self):
        numRandomEvalPoints = 5
        x = onp.random.uniform(low=0.0, high=1.0, size=(numRandomEvalPoints)) # random points on unit interval
        coeffs = onp.random.uniform(low=0.0, high=1.0, size=(3)) # random polynomial coefficients on unit interval

        element = Interpolants.make_lagrange_parent_element_1d(self.degree)
        p = element.coordinates
        _, dShape = Interpolants.shape1d_lagrange(element.degree, p, x)
        dfn = onp.polynomial.polynomial.polyval(p, coeffs)
        dfInterpolated = onp.dot(dfn, dShape)

        dPoly = onp.polynomial.polynomial.polyder(coeffs)
        expected = onp.polynomial.polynomial.polyval(x, dPoly)
        self.assertArrayNear(expected, np.array(dfInterpolated), 14)

    def test_tri_shape_kronecker_delta_property(self):
        element = Interpolants.make_lagrange_parent_element_2d(self.degree)
        shapeAtNodes, _ = Interpolants.shape2d_lagrange(element.degree, element.coordinates, element.coordinates)
        nNodes = element.coordinates.shape[0]
        self.assertArrayNear(shapeAtNodes, np.identity(nNodes), 14)

    def test_tri_shape_partition_of_unity(self):
        element = Interpolants.make_lagrange_parent_element_2d(self.degree)
        shapes, _ = Interpolants.shape2d_lagrange(element.degree, element.coordinates, self.qr.xigauss)
        self.assertArrayNear(np.sum(shapes, axis=1), np.ones(self.nQPts), 14)

    def test_tri_shapeGrads_partition_of_unity(self):
        element = Interpolants.make_lagrange_parent_element_2d(self.degree)
        _, shapeGradients = Interpolants.shape2d_lagrange(element.degree, element.coordinates, self.qr.xigauss)
        self.assertArrayNear(np.sum(shapeGradients, axis=1), np.zeros((self.nQPts, 2)), 14)

    def test_tri_interpolation(self):
        numRandomEvalPoints = 5
        element = Interpolants.make_lagrange_parent_element_2d(self.degree)
        self.assertCorrectInterpolationOn2DElement(element, numRandomEvalPoints, Interpolants.shape2d_lagrange)

    def test_tri_grad_interpolation(self):
        numRandomEvalPoints = 5
        element = Interpolants.make_lagrange_parent_element_2d(self.degree)
        self.assertCorrectGradInterpolationOn2DElement(element, numRandomEvalPoints, Interpolants.shape2d_lagrange)



if __name__ == '__main__':
    unittest.main()
