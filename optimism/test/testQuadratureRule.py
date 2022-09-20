from scipy.special import binom
import unittest

import jax.numpy as np

from optimism import QuadratureRule
from optimism.test import TestFixture


def is_inside_triangle(point):
    x_condition = (point[0] >= 0.) and (point[0] <= 1.)
    y_condition = (point[1] >= 0.) and (point[1] <= 1. - point[0])
    return (x_condition and y_condition)


def are_inside_unit_interval(points):
    return np.all(points >= 0.0) and np.all(points <= 1.0)


# integrate x^n y^m on unit triangle
def integrate_2D_monomial_on_triangle(n, m):
    p = n + m
    return 1.0/((p + 2)*(p + 1)*binom(p, n))


# integrate x^n on unit interval
def integrate_monomial_on_line(n):
    return 1.0/(n + 1)


def map_affine_1D(xi, endpoints):
    return (1.0 - xi)*endpoints[0] + xi*endpoints[1]


def map_1d_jac(endpoints):
    return (endpoints[1] - endpoints[0])


def are_positive_weights(QuadratureRuleFactory, degree):
    qr = QuadratureRuleFactory(degree)
    return np.all(qr.wgauss > 0)

    
class TestQuadratureRules(TestFixture.TestFixture):
    endpoints = (0.0, 1.0)     # better if the quadrature rule module provided this
    max_degree_2D = 6
    max_degree_1D = 25


    def test_1D_quadrature_weight_positivity(self):
        QuadratureRuleFactory = QuadratureRule.create_quadrature_rule_1D
        for degree in range(self.max_degree_1D + 1):
            self.assertTrue(are_positive_weights(QuadratureRuleFactory, degree))


    def test_1D_quadrature_points_in_domain(self):
        for degree in range(self.max_degree_1D + 1):
            quadrature_rule = QuadratureRule.create_quadrature_rule_1D(degree)
            self.assertTrue(are_inside_unit_interval(quadrature_rule.xigauss))


    def test_1D_quadrature_exactness(self):
        for degree in range(self.max_degree_1D + 1):
            qr = QuadratureRule.create_quadrature_rule_1D(degree)
            for i in range(degree + 1):
                map = lambda xi : map_affine_1D(xi, self.endpoints)
                xVals = map(qr.xigauss)
                jac = map_1d_jac(self.endpoints)
                monomial = xVals**i
                quadratureAnswer = np.sum(monomial * qr.wgauss)*jac
                exactAnswer = integrate_monomial_on_line(i)
                self.assertNear(quadratureAnswer, exactAnswer, 14)


    def test_triangle_quadrature_weight_positivity(self):
        for degree in range(self.max_degree_2D + 1):
            qr = QuadratureRule.create_quadrature_rule_on_triangle(degree)
            self.assertTrue(np.all(qr.wgauss > 0))


    def test_triangle_quadrature_points_in_domain(self):
        for degree in range(self.max_degree_2D + 1):
            qr = QuadratureRule.create_quadrature_rule_on_triangle(degree)
            for point in qr.xigauss:
                self.assertTrue(is_inside_triangle(point))


    def test_triangle_quadrature_exactness(self):
        for degree in range(self.max_degree_2D + 1):
            qr = QuadratureRule.create_quadrature_rule_on_triangle(degree)
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    monomial = qr.xigauss[:,0]**i * qr.xigauss[:,1]**j
                    quadratureAnswer = np.sum(monomial * qr.wgauss)
                    exactAnswer = integrate_2D_monomial_on_triangle(i, j)
                    self.assertNear(quadratureAnswer, exactAnswer, 14)
            
if __name__ == '__main__':
    unittest.main()
