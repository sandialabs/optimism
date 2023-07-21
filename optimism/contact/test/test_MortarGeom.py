from optimism.test.TestFixture import *
from optimism import QuadratureRule
from optimism.contact import MortarContact
import jax
import jax.numpy as np
import numpy as onp

usingTestingFilter = False


def spline_ramp(eta):
    # return 3*eta*eta - 2*eta*eta*eta # a less smoothing ramp
    #K = np.array([[3,4,5],[6,12,20],[1,1,1]])
    #r = np.array([0,0,1])
    #abc = np.linalg.solve(K,r)
    a = 10
    b = -15
    c = 6
    eta2 = eta*eta
    eta3 = eta2*eta
    return np.where(eta >= 1.0, 1.0, a * eta3 + b * eta2*eta2 + c * eta2*eta3)


def compute_error(edgeA, edgeB, xiA, xiB, g, f_common_normal):
    normal = f_common_normal(edgeA, edgeB)
    xA = MortarContact.eval_linear_field_on_edge(edgeA, xiA)
    xB = MortarContact.eval_linear_field_on_edge(edgeB, xiB)
    return np.linalg.norm(xA - xB + g * normal)


class TestMortarGeom(TestFixture):

    def setUp(self):
        self.f_common_normal = MortarContact.compute_average_normal #compute_normal_from_a

    @unittest.skipIf(usingTestingFilter, '')
    def testOffEdges(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.28], [0.15, -0.25]])
        xiA,xiB,g = MortarContact.compute_intersection(edgeA, edgeB, self.f_common_normal)
        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i], self.f_common_normal)
            self.assertTrue(err < 1e-15)


    @unittest.skipIf(usingTestingFilter, '')
    def testEdgesWithCommonPoint(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.3], [0.15, -0.25]])
        xiA,xiB,g = MortarContact.compute_intersection(edgeA, edgeB, self.f_common_normal)
        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i], self.f_common_normal)
            self.assertTrue(err < 1e-15)


    @unittest.skipIf(usingTestingFilter, '')
    def testEdgesWithTwoCommonPoints(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.3], [0.1, 0.2]])
        xiA,xiB,g = MortarContact.compute_intersection(edgeA, edgeB, self.f_common_normal)
        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i], self.f_common_normal)
            self.assertTrue(err < 1e-15)


    @unittest.skipIf(usingTestingFilter, '')
    def testAreaIntegrals(self):
        edgeA = np.array([[0.1, 0.1], [0.2, 0.1]])
        edgeB = np.array([[0.22, 0.0], [0.18, 0.0]])
        # eventually...
        # can give options on integrating on common vs different surfaces
        # can give options on quadrature rule on common surface
        commonArea = MortarContact.integrate_with_mortar(edgeA, edgeB, self.f_common_normal, lambda xiA, xiB, g: g)
        self.assertNear(commonArea, 0.002, 9)


    @unittest.skipIf(True, 'Used to explore smoothing')
    def testSpline(self):
        from matplotlib import pyplot as plt
        x = np.linspace(0.0,1.0,100)
        y = jax.vmap(spline_ramp)(x)
        #y = jax.vmap(smooth_linear,(0,None))(x,0.1)    
        plt.clf()
        plt.plot(x,y)
        plt.savefig('tmp.png')
    

    @unittest.skipIf(False, 'Used to explore contact behavior')
    def testMortarIntegralOneSided(self):
        from matplotlib import pyplot as plt

        def integrate_multipliers(edgeA, edgeB, lambdaA, lambdaB):
            xiThresh = 0.02
            def q(xiA, xiB, g):
                lamA = MortarContact.eval_linear_field_on_edge(lambdaA, xiA)
                lamB = MortarContact.eval_linear_field_on_edge(lambdaB, xiB)
                return g * (lamA + lamB)
            return MortarContact.integrate_with_mortar(edgeA, edgeB, self.f_common_normal, q, relativeSmoothingSize = 0.03)

        edgeA = np.array([[1.0, 0.1], [2.0, 0.1]])
        edgeB = np.array([[3.0, 0.0], [2.0, 0.0]])

        lambdaA = np.array([1.0, 1.0])
        lambdaB = np.zeros(2)

        def gap_of_y(y):
            ea = edgeA.at[0,0].set(y)
            ea = ea.at[1,0].set(y+1.0)
            return integrate_multipliers(ea, edgeB, lambdaA, lambdaB)

        N = 500
        edgeAy = np.linspace(0.9, 3.1, N)
        energy = jax.vmap(gap_of_y)(edgeAy)
        force = jax.vmap(jax.grad(gap_of_y))(edgeAy)

        plt.clf()
        plt.plot(edgeAy, energy)
        plt.savefig('energy.png')
        plt.clf()
        plt.plot(edgeAy, force)
        plt.savefig('force.png')


if __name__ == '__main__':
    unittest.main()