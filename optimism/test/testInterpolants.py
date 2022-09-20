import matplotlib.pyplot as plt
import numpy as onp
import unittest

import jax

from optimism import Interpolants
from optimism import QuadratureRule
from optimism.test import TestFixture

    
tol = 1e-14


class TestInterpolants(TestFixture.TestFixture):
    
    def setUp(self):
        maxDegree = 5
        self.masters1d = []
        self.mastersTri = []
        self.qr1d = QuadratureRule.create_quadrature_rule_1D(degree=maxDegree)
        self.nQPts1d = QuadratureRule.len(self.qr1d)
        self.qr = QuadratureRule.create_quadrature_rule_on_triangle(degree=maxDegree)
        self.nQPts = QuadratureRule.len(self.qr)
        for degree in range(1, maxDegree + 1):
            basis1d = Interpolants.make_nodal_basis_1d(degree)
            basis = Interpolants.make_nodal_basis_2d(degree)
            self.masters1d.append(Interpolants.make_master_line_element(basis1d, self.qr1d))
            self.mastersTri.append(Interpolants.make_master_tri_element(basis, self.qr))


    def test_1D_interpolant_points_in_element(self):
        for master in self.masters1d:
            p = master.coordinates
            self.assertTrue(onp.all(p >= 0.0) and onp.all(p <= 1.0))


    def test_1D_master_element_topological_nodesets(self):
        for master in self.masters1d:
            p = master.coordinates
            self.assertNear(p[master.vertexNodes[0]], 0.0, 14)
            self.assertNear(p[master.vertexNodes[1]], 1.0, 14)
            
            if master.interiorNodes is not None:
                self.assertTrue(onp.all(p[master.interiorNodes] > 0.0))
                self.assertTrue(onp.all(p[master.interiorNodes] < 1.0))

            
    def test_tri_interpolant_points_in_element(self):
        for master in self.mastersTri:
            p = master.coordinates
            # x conditions
            self.assertTrue(onp.all(p[:,0] >= -tol))
            self.assertTrue(onp.all(p[:,0] <= 1.0 + tol))
            # y conditions
            self.assertTrue(onp.all(p[:,1] >= -tol))
            self.assertTrue(onp.all(p[:,1] <= 1. - p[:,0] + tol))


    def test_tri_master_element_topological_nodesets(self):
        for master in self.mastersTri:
            p = master.coordinates
            self.assertArrayNear(p[master.vertexNodes[0],:],
                                 onp.array([1.0, 0.0]),
                                 14)
            self.assertArrayNear(p[master.vertexNodes[1],:],
                                 onp.array([0.0, 1.0]),
                                 14)
            self.assertArrayNear(p[master.vertexNodes[2],:],
                                 onp.array([0.0, 0.0]),
                                 14)
            
            if master.interiorNodes.size > 0:
                k = master.interiorNodes
                self.assertTrue(onp.all(p[k,0] > -tol))
                self.assertTrue(onp.all(p[k,1] + p[k,0] - 1. <  tol))


    def test_tri_face_nodes_match_1D_lobatto_nodes(self):
        for master1d, masterTri in zip(self.masters1d, self.mastersTri):
            for faceNodeIds in masterTri.faceNodes:
                # get the triangle face node points directly
                xf = masterTri.coordinates[faceNodeIds,:]
                # affine transformation of 1D node points to triangle face
                p = master1d.coordinates
                x1d = onp.outer(1.0 - p, xf[0,:]) + onp.outer(p, xf[-1,:])
                # make sure they are the same
                self.assertArrayNear(xf, x1d, 14)


    def test_tri_shape_partition_of_unity(self):
        for master in self.mastersTri:
            self.assertArrayNear(onp.sum(master.shapes, axis=1), onp.ones(self.nQPts), 14)


    def test_tri_shapeGrads_partition_of_unity(self):
        for master in self.mastersTri:
            self.assertArrayNear(onp.sum(master.shapeGradients, axis=1), onp.zeros((self.nQPts, 2)), 14)

            
    def test_shape_kronecker_delta_property(self):
        for master in self.mastersTri:
            shapeAtNodes, _ = Interpolants.shape2d(master.degree, master.coordinates, master.coordinates)
            nNodes = master.coordinates.shape[0]
            self.assertArrayNear(shapeAtNodes, onp.identity(nNodes), 14)


    def test_interpolation(self):
        x = self.generate_random_points_in_triangle(1)
        for master in self.mastersTri:
            degree = master.degree
            polyCoeffs = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))
            expected = onp.polynomial.polynomial.polyval2d(x[:,0], x[:,1], polyCoeffs)

            shape, _ = Interpolants.shape2d(degree, master.coordinates, x)
            fn = onp.polynomial.polynomial.polyval2d(master.coordinates[:,0],
                                                     master.coordinates[:,1],
                                                     polyCoeffs)
            fInterpolated = onp.dot(shape, fn)
            self.assertArrayNear(expected, fInterpolated, 14)


    def test_grad_interpolation(self):
        x = self.generate_random_points_in_triangle(1)
        for master in self.mastersTri:
            degree = master.degree
            poly = onp.fliplr(onp.triu(onp.ones((degree+1,degree+1))))

            _, dShape = Interpolants.shape2d(degree, master.coordinates, x)
            fn = onp.polynomial.polynomial.polyval2d(master.coordinates[:,0],
                                                     master.coordinates[:,1],
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
        y = jax.numpy.zeros(npts)
        for i in range(npts):
            key,subkey = jax.random.split(key)
            y = y.at[i].set(jax.random.uniform(subkey, minval=0.0, maxval=1.0-x[i]))

        points = jax.numpy.column_stack((x,y))
        return  onp.asarray(points)


# class TestBubbleInterpolants(TestInterpolants):
#     def setUp(self):
#         maxDegree = 3
#         self.masters1d = []
#         self.mastersTri = []
#         for degree in range(2, maxDegree + 1):
#             self.masters1d.append(Interpolants.make_master_line_element(degree))
#             self.mastersTri.append(Interpolants.make_master_tri_bubble_element(degree))


#     def no_test_plot_shape_functions(self):
#         s = self.generate_random_points_in_triangle(500)
#         shape = Interpolants.compute_shapes_on_bubble_tri(self.mastersTri[1], s)
#         unit = np.sum(shape, axis=1)
#         print(np.max(unit), np.min(unit))

#         for i in range(Interpolants.num_nodes(self.mastersTri[1])):
#             ax = plt.figure().add_subplot(projection='3d')
#             ax.scatter(s[:,0], s[:,1], shape[:,i])

#         plt.show()


if __name__ == '__main__':
    unittest.main()
