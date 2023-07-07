from optimism.test.TestFixture import *
from optimism import QuadratureRule
import jax
import jax.numpy as np
import numpy as onp

usingTestingFilter = False


def smooth_linear(xi, l):                
    return np.where( xi < l, 0.5*xi*xi/l, np.where(xi > 1.0-l, 1.0-l-0.5*(1.0-xi)*(1.0-xi)/l, xi-0.5*l) )


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


def eval_linear_field_on_edge(field, xi):
    return field[0] * (1.0 - xi) + field[1] * xi


def compute_normal(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = np.array([tangent[1], -tangent[0]])
    return normal / np.linalg.norm(normal)


def compute_average_normal(edgeA, edgeB):
    nA = compute_normal(edgeA)
    nB = compute_normal(edgeB)
    normal = nA - nB
    return normal / np.linalg.norm(normal)


def compute_normal_from_a(edgeA, edgeB):
    return compute_normal(edgeA)


def compute_error(edgeA, edgeB, xiA, xiB, g, f_common_normal):
    normal = f_common_normal(edgeA, edgeB)
    xA = eval_linear_field_on_edge(edgeA, xiA)
    xB = eval_linear_field_on_edge(edgeB, xiB)
    return np.linalg.norm(xA - xB + g * normal)


def compute_intersection(edgeA, edgeB, f_common_normal):

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = np.array([edgeB[0]-edgeB[1], normal]).T
        r = np.array(edgeB[0]-xa)
        xig = np.linalg.solve(M,r)
        return xig[0], xig[1]

    normal = f_common_normal(edgeA, edgeB)
    xiBs1, gs1 = jax.vmap(compute_xi, (0,None,None))(edgeA, edgeB, normal)
    xiAs2, gs2 = jax.vmap(compute_xi, (0,None,None))(edgeB, edgeA,-normal)

    xiAs = np.hstack((np.arange(2), xiAs2))
    xiBs = np.hstack((xiBs1, np.arange(2)))
    gs = np.hstack((gs1, gs2))

    xiAgood = jax.vmap(lambda xia, xib: np.where((xia >= 0.0) & (xia <= 1.0) & (xib >= 0.0) & (xib <= 1.0), xia, np.nan))(xiAs, xiBs)
    argsMinMax = np.array([np.nanargmin(xiAgood), np.nanargmax(xiAgood)])

    return xiAs[argsMinMax], xiBs[argsMinMax], gs[argsMinMax]


def integrate_with_active_mortar(xiA, xiB, g, lengthA, lengthB, func_of_xiA_xiB_w_g):
    edgeQuad = QuadratureRule.create_quadrature_rule_1D(degree=2)
    xiGauss = edgeQuad.xigauss

    quadXiA = jax.vmap(eval_linear_field_on_edge, (None,0))(xiA, xiGauss)
    quadXiB = jax.vmap(eval_linear_field_on_edge, (None,0))(xiB, xiGauss)

    smoothingSize = 0.1
    xiAsmooth = smooth_linear(xiA, smoothingSize)
    xiBsmooth = smooth_linear(xiB, smoothingSize)
    dxiA = xiAsmooth[1] - xiAsmooth[0]
    dxiB = np.abs(xiBsmooth[1] - xiBsmooth[0])

    quadWeightA = lengthA * dxiA * edgeQuad.wgauss
    quadWeightB = lengthB * dxiB * edgeQuad.wgauss
    gs = jax.vmap(eval_linear_field_on_edge, (None,0))(g, xiGauss)

    return np.sum(jax.vmap(func_of_xiA_xiB_w_g)(quadXiA, quadXiB, 0.5*(quadWeightA+quadWeightB), gs))


def integrate_with_mortar(edgeA, edgeB, f_common_normal, func_of_xiA_xiB_w_g):
    xiA,xiB,g = compute_intersection(edgeA, edgeB, f_common_normal)
    branches = [lambda : integrate_with_active_mortar(xiA, xiB, g, 
                                                      np.linalg.norm(edgeA[0] - edgeA[1]), 
                                                      np.linalg.norm(edgeB[0] - edgeB[1]),
                                                      func_of_xiA_xiB_w_g),
                lambda : 0.0]

    return jax.lax.switch(1*np.any(xiA==np.nan), branches)


class TestMortarGeom(TestFixture):

    def setUp(self):
        self.f_common_normal = compute_average_normal #compute_normal_from_a

    @unittest.skipIf(usingTestingFilter, '')
    def testOffEdges(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.28], [0.15, -0.25]])
        xiA,xiB,g = compute_intersection(edgeA, edgeB, self.f_common_normal)
        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i], self.f_common_normal)
            self.assertTrue(err < 1e-15)

    @unittest.skipIf(usingTestingFilter, '')
    def testEdgesWithCommonPoint(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.3], [0.15, -0.25]])
        xiA,xiB,g = compute_intersection(edgeA, edgeB, self.f_common_normal)
        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i], self.f_common_normal)
            self.assertTrue(err < 1e-15)

    @unittest.skipIf(usingTestingFilter, '')
    def testEdgesWithTwoCommonPoints(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.3], [0.1, 0.2]])
        xiA,xiB,g = compute_intersection(edgeA, edgeB, self.f_common_normal)
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
        commonArea = integrate_with_mortar(edgeA, edgeB, self.f_common_normal, lambda xiA, xiB, w, g: w*g)
        self.assertNear(commonArea, 0.002, 16)

    unittest.skipIf(True, '')
    def testSpline(self):
        from matplotlib import pyplot as plt
        x = np.linspace(0.0,1.0,100)
        y = jax.vmap(spline_ramp)(x)
        #y = jax.vmap(smooth_linear,(0,None))(x,0.1)    
        plt.clf()
        plt.plot(x,y)
        plt.savefig('tmp.png')
    
    unittest.skipIf(True, '')
    def testMortarIntegralOneSided(self):
        from matplotlib import pyplot as plt

        def integrate_multipliers(edgeA, edgeB, lambdaA, lambdaB):
            xiThresh = 0.02
            def q(xiA, xiB, w, g):
                lamA = eval_linear_field_on_edge(lambdaA, xiA)
                lamB = eval_linear_field_on_edge(lambdaB, xiB)
                return w * g * (lamA + lamB)

            return integrate_with_mortar(edgeA, edgeB, self.f_common_normal, q)

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