from optimism.test.TestFixture import *
from optimism import QuadratureRule
import jax
import jax.numpy as np

usingTestingFilter = True


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
    normal = compute_common_normal(edgeA, edgeB)

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = np.array([edgeB[0]-edgeB[1], normal]).T
        r = np.array(edgeB[0]-xa)
        xig = np.linalg.solve(M,r)
        return xig[0], xig[1]

    xiAs = []
    xiBs = []
    gs = []

    for i in range(2):
        xiA = 1.0 * (i==1)
        xiB, g = compute_xi(edgeA[i], edgeB, normal)
        xiAs.append(xiA)
        xiBs.append(xiB)
        gs.append(g)

    for i in range(2):
        xiB = 1.0 * (i==1)
        xiA, g = compute_xi(edgeB[i], edgeA, -normal)
        xiAs.append(xiA)
        xiBs.append(xiB)
        gs.append(g)
        
    xiAk = [] # k is for korrected
    xiBk = []
    gk = []
    for i in range(len(xiAs)):
        xiA = xiAs[i]
        xiB = xiBs[i]
        if xiA <= 1.0 and xiA >= 0.0 and xiB <= 1.0 and xiB >= 0.0:
            xiAk.append(xiA)
            xiBk.append(xiB)
            gk.append(gs[i])

    xiAk = np.array(xiAk)
    xiBk = np.array(xiBk)
    gk = np.array(gk)

    if len(gk) > 2:
        xiAk = np.zeros(2)
        xiBk = np.zeros(2)
        gk = np.zeros(2)

        xiAMin = np.min(xiAk)
        xAtMin = eval_linear_field_on_edge(edgeA, xiAMin)
        xiBMin, gMin = compute_xi(xAtMin, edgeB, normal)
        xiAk = xiAk.at[0].set(xiAMin)
        xiBk = xiBk.at[0].set(xiBMin)
        gk = gk.at[0].set(gMin)

        xiAMax = np.max(xiAk)
        xAtMax = eval_linear_field_on_edge(edgeA, xiAMax)
        xiBMax, gMax = compute_xi(xAtMax, edgeB, normal)
        xiAk = xiAk.at[1].set(xiAMax)
        xiBk = xiBk.at[1].set(xiBMax)
        gk = gk.at[1].set(gMax)

    if len(gk)==0:
        return None,None,None
    
    if len(gk)==1 or len(gk) > 2:
        print("\n\nA very unusual bug found in intersection test")
        print("edges = \n\n", edgeA, edgeB)
        return None, None, None

    return xiAk, xiBk, gk
        

def create_mortar_integrator(edgeA, edgeB):
    xiA,xiB,g = compute_intersection(edgeA, edgeB)
    if xiA == None:
        return lambda func : 0.0

    edgeQuad = QuadratureRule.create_quadrature_rule_1D(degree=2)
    xiGauss = edgeQuad.xigauss

    quadXiA = jax.vmap(eval_linear_field_on_edge, (None,0))(xiA, xiGauss)
    lengthA = np.linalg.norm(edgeA[0] - edgeA[1])
    quadWeightA = lengthA * np.abs(xiA[0]-xiA[1]) * edgeQuad.wgauss

    quadXiB = jax.vmap(eval_linear_field_on_edge, (None,0))(xiB, xiGauss)
    lengthB = np.linalg.norm(edgeB[0] - edgeB[1])
    quadWeightB = lengthB * np.abs(xiB[0]-xiB[1]) * edgeQuad.wgauss

    gs = jax.vmap(eval_linear_field_on_edge, (None,0))(g, xiGauss)

    if (np.linalg.norm(quadWeightA-quadWeightB) > 1e-14):
        print("\n\nPoor accuracy for mortar integrals, the two side disagree on their common area")
        print("edges = \n\n", edgeA, edgeB)

    def integrator(func_of_xiA_xiB_g_weight):
        integral = jax.vmap(func_of_xiA_xiB_g_weight, (0,0,0,0))(quadXiA, quadXiB, quadWeightA, gs)
        return np.sum(integral)

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

    #@unittest.skipIf(usingTestingFilter, '')
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


if __name__ == '__main__':
    unittest.main()