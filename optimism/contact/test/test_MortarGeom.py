from optimism.test.TestFixture import *
from optimism import QuadratureRule
import jax
import jax.numpy as np
import numpy as onp

usingTestingFilter = False

def eval_linear_field_on_edge(field, xi):
    return field[0] * (1.0 - xi) + field[1] * xi


def compute_normal(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = np.array([tangent[1], -tangent[0]])
    return normal / np.linalg.norm(normal)


def compute_common_normal(edgeA, edgeB):
    nA = compute_normal(edgeA)
    nB = compute_normal(edgeB)
    normal = nA - nB
    return normal / np.linalg.norm(normal)


def compute_error(edgeA, edgeB, xiA, xiB, g):
    normal = compute_common_normal(edgeA, edgeB)
    xA = eval_linear_field_on_edge(edgeA, xiA)
    xB = eval_linear_field_on_edge(edgeB, xiB)
    return np.linalg.norm(xA - xB + g * normal)


def compute_intersection(edgeA, edgeB):

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = np.array([edgeB[0]-edgeB[1], normal]).T
        r = np.array(edgeB[0]-xa)
        xig = np.linalg.solve(M,r)
        return xig[0], xig[1]

    normal = compute_common_normal(edgeA, edgeB)
    xiBs1, gs1 = jax.vmap(compute_xi, (0,None,None))(edgeA, edgeB, normal)
    xiAs2, gs2 = jax.vmap(compute_xi, (0,None,None))(edgeB, edgeA, -normal)

    xiAs = np.hstack((np.arange(2), xiAs2))
    xiBs = np.hstack((xiBs1, np.arange(2)))
    gs = np.hstack((gs1, gs2))

    xiAgood = jax.vmap(lambda xia, xib: np.where((xia >= 0.0) & (xia <= 1.0) & (xib >= 0.0) & (xib <= 1.0), xia, np.nan))(xiAs, xiBs)
    argsMinMax = np.array([np.nanargmin(xiAgood), np.nanargmax(xiAgood)])

    return xiAs[argsMinMax], xiBs[argsMinMax], gs[argsMinMax]


def create_mortar_integrator(edgeA, edgeB):
    xiA,xiB,g = compute_intersection(edgeA, edgeB)

    edgeQuad = QuadratureRule.create_quadrature_rule_1D(degree=2)
    xiGauss = edgeQuad.xigauss

    quadXiA = jax.vmap(eval_linear_field_on_edge, (None,0))(xiA, xiGauss)
    lengthA = np.linalg.norm(edgeA[0] - edgeA[1])
    quadWeightA = lengthA * np.abs(xiA[0]-xiA[1]) * edgeQuad.wgauss

    quadXiB = jax.vmap(eval_linear_field_on_edge, (None,0))(xiB, xiGauss)
    #lengthB = np.linalg.norm(edgeB[0] - edgeB[1])
    #quadWeightB = lengthB * np.abs(xiB[0]-xiB[1]) * edgeQuad.wgauss

    gs = jax.vmap(eval_linear_field_on_edge, (None,0))(g, xiGauss)

    #if (np.linalg.norm(quadWeightA-quadWeightB) > 1e-14):
    #    print("\n\nPoor accuracy for mortar integrals, the two side disagree on their common area")
    #    print("edges = \n\n", edgeA, edgeB)

    def integrator(func_of_xiA_xiB_weight_g): # returns the integral of the provided kernel ('q') function over the intersection of edges
        branches = [func_of_xiA_xiB_weight_g, lambda xiA, xiB, w, g: 0.0]
        def integrand(xiA, xiB, w, g):
            index = np.where(xiA==np.nan, 1, 0)
            return jax.lax.switch(index, branches, xiA, xiB, w, g)
                
        return np.sum(jax.vmap(integrand)(quadXiA, quadXiB, quadWeightA, gs))

    return integrator


class TestMortarGeom(TestFixture):
    
    def setUp(self):
        return

    @unittest.skipIf(usingTestingFilter, '')
    def testOffEdges(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.28], [0.15, -0.25]])
        xiA,xiB,g = compute_intersection(edgeA, edgeB)

        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i])
            self.assertTrue(err < 1e-15)

    @unittest.skipIf(usingTestingFilter, '')
    def testEdgesWithCommonPoint(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.3], [0.15, -0.25]])
        xiA,xiB,g = compute_intersection(edgeA, edgeB)

        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i])
            self.assertTrue(err < 1e-15)

    @unittest.skipIf(usingTestingFilter, '')
    def testEdgesWithTwoCommonPoints(self):
        edgeA = np.array([[0.1, 0.2], [-0.2, 0.3]])
        edgeB = np.array([[-0.2, 0.3], [0.1, 0.2]])
        xiA,xiB,g = compute_intersection(edgeA, edgeB)

        for i in range(len(g)):
            err = compute_error(edgeA, edgeB, xiA[i], xiB[i], g[i])
            self.assertTrue(err < 1e-15)

    @unittest.skipIf(usingTestingFilter, '')
    def testAreaIntegrals(self):
        edgeA = np.array([[0.1, 0.1], [0.2, 0.1]])
        edgeB = np.array([[0.22, 0.0], [0.18, 0.0]])
        # eventually...
        # can give options about how to compute common normal
        # can give options on integrating on common vs different surfaces
        # can give options on quadrature rule on common surface
        edgeIntegrator = create_mortar_integrator(edgeA, edgeB)
        commonArea = edgeIntegrator(lambda xiA, xiB, w, g: w*g)
        self.assertNear(commonArea, 0.002, 16)

    def testMortarIntegralOneSided(self):
        
        from matplotlib import pyplot as plt

        @jax.jit
        def integrate_area(edgeA, edgeB, lambdaA, lambdaB):
            edgeIntegrator = create_mortar_integrator(edgeA, edgeB)

            # 1-sided integral for now
            def q(xiA, xiB, w, g):
                lamA = eval_linear_field_on_edge(lambdaA, xiA)
                #print('shape lamA = ', lamA)
                #return w * g
                #lamB = eval_linear_field_on_edge(lambdaB, xiB)
                return w * lamA * g 

            return edgeIntegrator(q)
        
        edgeA = np.array([[0.1, 0.1], [0.2, 0.1]])
        edgeB = np.array([[0.22, 0.0], [0.18, 0.0]])
        lambdaA = np.array([0.0, 1.0])
        lambdaB = np.zeros(2)

        N = 101
        edgeAy = np.linspace(0.09, 0.11, N)
        energy = onp.zeros(len(edgeAy))

        print("e1 = ", integrate_area(edgeA, edgeB, lambdaA, lambdaB))

        #for n in range(N):
        #    energy[n] = integrate_area( edgeA.at[0,1].set(edgeAy[n]), edgeB, lambdaA, lambdaB )

        plt.plot(edgeAy, energy)
        plt.savefig('plt.png')


if __name__ == '__main__':
    unittest.main()