from scipy.special import legendre, jacobi
from jax.numpy.linalg import solve
import numpy as onp

from optimism.JaxConfig import *


MasterElement = namedtuple('MasterElement',
                           ['degree', 'coordinates', 'vertexNodes', 'faceNodes', 'interiorNodes'])


MasterBubbleElement = namedtuple('MasterBubbleElement',
                                 ['degree', 'coordinates', 'vertexNodes', 'faceNodes', 'interiorNodes', 'baseMaster', 'baseMasterNodes', 'enrichmentMaster', 'enrichmentMasterNodes'])


def make_master_elements(degree):
    master = make_master_tri_element(degree)
    master1d = make_master_line_element(degree)
    return master, master1d


def make_master_line_element(degree):
    """Gauss-Lobatto Interpolation points on the unit interval.
    """
    # define on standard [-1, 1]
    xInterior = np.array([])
    if degree == 0:
        xn = np.array([0.0])
        vertexPoints = np.array([], dtype=np.int32)
        interiorPoints = np.array([1], dtype=np.int32)
    else:
        P = legendre(degree)
        P = np.array(P.c) # convert to jax deviceArray
        DP = np.polyder(P)
        xInterior = np.real(np.sort(np.roots(DP)))
        xn = np.hstack( (np.array([-1.0]), xInterior, np.array([1.0])) )
        vertexPoints = np.array([0, degree], dtype=np.int32)
        interiorPoints = np.arange(1, degree)

    #  shift to [0, 1]
    xn = 0.5*(xn + 1.0)
    
    return MasterElement(degree, xn, vertexPoints, None, interiorPoints)


def make_master_tri_element(degree):
    """Interpolation points on the triangle that are Lobatto points on the edges.
    Points have threefold rotational symmetry and low Lebesgue constants.
    
    Reference:
    M.G. Blyth and C. Pozrikidis. "A Lobatto interpolation grid over the triangle"
    IMA Journal of Applied Mathematics (2005) 1-17.
    doi:10.1093/imamat/hxh077
    
    Convention for numbering:
    example: degree = 3
    
       3
       o
       | \
     6 o  o 2    interior node is 5
       |    \
     8 o  o  o 1
       |       \
       o--o--o--o
       9  7  4   0
    """

    if degree == 0:
        points = np.array([1.0, 1.0])/3.0
        vertexPoints = np.array([0], dtype=np.int32)
        facePoints = np.array([[0],[0],[0]], dtype=np.int32)
        interiorPoints = np.array([0], dtype=np.int32)
    else:
        lobattoPoints = make_master_line_element(degree).coordinates
        nPoints = int((degree + 1)*(degree + 2)/2)
        points = np.zeros((nPoints, 2))
        point = 0
        for i in range(degree):
            for j in range(degree + 1 - i):
                k = degree - i - j
                points = points.at[ (point,0) ].set( (1.0 + 2.0*lobattoPoints[k] - lobattoPoints[j] - lobattoPoints[i])/3.0 )
                points = points.at[ (point,1) ].set( (1.0 + 2.0*lobattoPoints[j] - lobattoPoints[i] - lobattoPoints[k])/3.0 )
                point += 1
    
        vertexPoints = np.array([0, degree, nPoints - 1], dtype=int)

        ii = np.arange(degree + 1)
        jj = np.cumsum(np.flip(ii)) + ii
        kk = np.flip(jj) - ii
        facePoints = np.array((ii,jj,kk), dtype=int)

        interiorPoints = [i for i in range(nPoints) if i not in facePoints.ravel()]
        interiorPoints = np.array(interiorPoints, dtype=int)
    
    return MasterElement(degree, points, vertexPoints, facePoints, interiorPoints)
    
    
def make_vandermonde_1D(x, degree):
    # shift to bi-unit interval convention of scipy
    z = 2.0*x - 1.0

    A = []
    for i in range(degree + 1):
        P = legendre(i)
        P = np.array(P.c)
        # re-scale back to [0,1]
        P *= np.sqrt(2.0*i + 1.0)
        Pval = np.polyval(P, z)
        A.append(Pval)

    return np.array(A).T


def compute_1D_shape_function_values(nodalPoints, evaluationPoints, degree):
    """Evalute 1D shape functions at points in the master element.
    
    shapes: [nNodes, nEvalPoints]
    """
    A = make_vandermonde_1D(nodalPoints, degree)
    nf = make_vandermonde_1D(evaluationPoints, degree)
    return solve(A.T, nf.T) 


compute_1D_shape_function_derivative_values = jacfwd(compute_1D_shape_function_values, 1)


def compute_shapes_1D(nodalPoints, evaluationPoints, degree):
    return vmap(make_1D_shape_function_values, (None, 0, None))(nodalPoints, evaluationPoints, degree)


def compute_dshapes_1D(nodalPoints, evaluationPoints, degree):
    return vmap(make_1D_shape_function_derivative_values, (None, 0, None))(nodalPoints, evaluationPoints, degree)


def pascal_triangle_monomials(degree):
    p = []
    q = []
    for i in range(1, degree + 2):
        monomialIndices = [j for j in range(i)]
        p += monomialIndices
        monomialIndices.reverse()
        q += monomialIndices
    return np.column_stack((q,p))


def compute_vandermonde_tri(x, degree):
    assert np.issubdtype(type(degree), np.integer)
    nNodes = (degree+1)*(degree+2)//2
    pq = pascal_triangle_monomials(degree)

    # It's easier to process if the input arrays
    # always have the same shape
    # If a 1D array is given (a single point),
    # convert to the equivalent 2D array
    x = x.reshape(-1,2)
    
    # switch to bi-unit triangle (-1,-1)--(1,-1)--(-1,1)
    z = 2.0*x - 1.0

    # now map onto bi-unit square
    def map_from_tri_to_square(point):
        singularPoint = np.array([-1.0, 1.0])
        pointIsSingular = np.array_equal(point, singularPoint)
        point = np.where(pointIsSingular, np.array([0.0, 0.0]), point)
        newPoint = np.where(pointIsSingular,
                            np.array([-1.0, 1.0]),
                            np.array([2.0*(1.0 + point[0])/(1.0 - point[1]) - 1.0, point[1]]))
        return newPoint
    E = vmap(map_from_tri_to_square)(z)
    
    A = np.zeros((x.shape[0], nNodes))
    for i in range(nNodes):
        PP = legendre(pq[i,0])
        PP = np.array(PP.c) # convert from scipy polynomial to devicearray
        QP = jacobi(pq[i,1], 2*pq[i,0] + 1.0, 0)
        QP = np.array(QP.c)
        for j in range(pq[i,0]):
            QP = np.polymul(np.array([-0.5,0.5]),QP)
        pVal = np.polyval(PP, E[:,0])
        qVal = np.polyval(QP, E[:,1])
        weight = np.sqrt( (2*pq[i,0]+1) * 2*(pq[i,0]+pq[i,1]+1))
        A = A.at[:,i].set(weight*pVal*qVal)
    return np.array(A)


def compute_shapes_on_tri(masterElement, evaluationPoints):
    # BT: The fake polymorphism here is not satisfying or
    # extensible. We could probably make the master
    # elements into classes. We just want to be sure
    # we're ok with them being types that can't be
    # registered in jax.
    
    if type(masterElement) == MasterElement:
        shapes =  _compute_shapes_on_tri(masterElement,
                                         evaluationPoints)
    elif type(masterElement) == MasterBubbleElement:
        shapes = _compute_shapes_on_bubble_tri(masterElement,
                                               evaluationPoints)
    else:
        raise TypeError('Unrecognized master element type')

    return shapes


def _compute_shapes_on_tri(masterElement, evaluationPoints):
    A = compute_vandermonde_tri(masterElement.coordinates, masterElement.degree)
    nf = compute_vandermonde_tri(evaluationPoints, masterElement.degree)
    return solve(A.T, nf.T).T


compute_dshape_on_tri = jacfwd(compute_shapes_on_tri, 1)


def compute_shapeGrads_on_tri(masterElement, evaluationPoints):
    f = lambda m,x: np.squeeze(compute_dshape_on_tri(m,x))
    return vmap(f, (None,0))(masterElement, evaluationPoints)


def make_master_tri_bubble_element(degree):
    baseMaster = make_master_tri_element(degree)
    bubbleMaster = make_master_tri_element(degree + 1)

    nPointsBase = num_nodes(baseMaster) - baseMaster.interiorNodes.size
    nPointsBubble = bubbleMaster.interiorNodes.shape[0]
    nPoints = nPointsBase + nPointsBubble

    nNodesBase = num_nodes(baseMaster)
    baseNonInteriorNodes = np.full(nNodesBase, True).at[baseMaster.interiorNodes].set(False)

    coords = np.zeros((nPoints,2)).at[:nPointsBase].set(baseMaster.coordinates[baseNonInteriorNodes])
    coords = coords.at[nPointsBase:].set(bubbleMaster.coordinates[bubbleMaster.interiorNodes])

    vertexNodes = np.array([0, degree , nPointsBase - 1], dtype=int)

    ii = np.arange(degree + 1)
    jj = np.array([i for i in range(degree, 3*degree, 2)] + [nPointsBase - 1], dtype=int)
    kk = np.array([i for i in reversed(range(degree + 1, nPointsBase, 2))] + [0], dtype=int)
    faceNodes = np.array((ii,jj,kk), dtype=int)

    interiorNodes =  np.arange(nPointsBase, nPointsBase + nPointsBubble)

    baseMasterNodes = np.setdiff1d(np.arange(num_nodes(baseMaster)),
                                   baseMaster.interiorNodes,
                                   assume_unique=True)

    bubbleMasterNodes = bubbleMaster.interiorNodes

    return MasterBubbleElement(degree, coords, vertexNodes, faceNodes,
                               interiorNodes, baseMaster, baseMasterNodes,
                               bubbleMaster, bubbleMasterNodes)


def _compute_shapes_on_bubble_tri(master, evaluationPoints):
    # base shape function values at eval points
    baseMaster = master.baseMaster
    baseShapes = _compute_shapes_on_tri(baseMaster,
                                        evaluationPoints)
    baseShapes = baseShapes[:,master.baseMasterNodes]

    # base shape functions at bubble nodes
    baseShapesAtBubbleNodes = _compute_shapes_on_tri(baseMaster,
                                                     master.coordinates[master.interiorNodes])
    baseShapesAtBubbleNodes = baseShapesAtBubbleNodes[:,master.baseMasterNodes]

    # bubble function values at eval points
    enrichmentMaster = master.enrichmentMaster
    bubbleShapes = _compute_shapes_on_tri(enrichmentMaster,
                                          evaluationPoints)
    bubbleShapes = bubbleShapes[:,master.enrichmentMasterNodes]

    return np.hstack((baseShapes - bubbleShapes@baseShapesAtBubbleNodes,
                      bubbleShapes))


def num_nodes(master):
    return master.coordinates.shape[0]
