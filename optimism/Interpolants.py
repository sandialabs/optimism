from collections import namedtuple
import numpy as onp
from scipy import special

import jax.numpy as np


ParentElement = namedtuple('NodalBasis',
                        ['elementType', 'degree', 'coordinates', 'vertexNodes', 'faceNodes', 'interiorNodes'])
ParentElement.__doc__ = \
    """Finite element on reference domain.

    Attributes:
        elementType: integer indiacting the element type. The magic numbers are
            defined in the ``Interpolants`` module.
        degree: Highest degree complete polynomial the element is capable of
            exactly representing.
        coordinates: Locations of the nodes in parameteric coordinates. Rows
            are nodes, columns are x and y.
        vertexNodes: Indices of vertices in the ``coordinates`` array.
        faceNodes: Indices of the nodes on each triangle face in the
            ``coordinates`` array. For example, ``faceNodes[0]`` gives the
            indices of the nodes on the face between vertex 0 and vertex 1.
            This is empty for line elements.
        interiorNodes: Indices of nodes that are not on the boundary of the
            element.
    """

ShapeFunctions = namedtuple('ShapeFunctions', ['values', 'gradients'])
ShapeFunctions.__doc__ = \
    """Shape functions and shape function gradients (in the parametric space).

    Attributes:
        values: Values of the shape functions at a discrete set of points.
            Shape is ``(nPts, nNodes)``, where ``nPts`` is the number of
            points at which the shame functinos are evaluated, and ``nNodes``
            is the number of nodes in the element (which is equal to the
            number of shape functions).
        gradients: Values of the parametric gradients of the shape functions.
            Shape is ``(nPts, nDim, nNodes)``, where ``nDim`` is the number
            of spatial dimensions. Line elements are an exception, which
            have shape ``(nPts, nNdodes)``.
    """

# element types
LINE_ELEMENT = 0
TRIANGLE_ELEMENT = 1
TRIANGLE_ELEMENT_WITH_BUBBLE = 2


def make_parent_elements(degree):
    """Returns a triangle element and the corresponding line element."""
    basis = make_parent_element_2d(degree)
    basis1d = make_parent_element_1d(degree)
    return basis, basis1d


def make_parent_element_1d(degree):
    """Gauss-Lobatto Interpolation points on the unit interval.
    """

    xn = get_lobatto_nodes_1d(degree)
    vertexPoints = np.array([0, degree], dtype=np.int32)
    interiorPoints = np.arange(1, degree, dtype=np.int32)
    return ParentElement(LINE_ELEMENT, int(degree), xn, vertexPoints, None, interiorPoints)


def get_lobatto_nodes_1d(degree):
    p = onp.polynomial.Legendre.basis(degree, domain=[0.0, 1.0])
    dp = p.deriv()
    xInterior = dp.roots()
    xn = np.hstack(([0.0], xInterior, [1.0]))
    return xn


def shape1d(degree, nodalPoints, evaluationPoints):
    """Evaluate shape functions and derivatives at points in the master element.

    Args:
      master1d: 1D MasterElement to evaluate the shape function data on
      evaluationPoints: Array of points in the master element domain at
        which to evaluate the shape functions and derivatives.

    Returns:
      Shape function values and shape function derivatives at ``evaluationPoints``,
      in a tuple (``shape``, ``dshape``).
      shapes: [nNodes, nEvalPoints]
      dshapes: [nNodes, nEvalPoints]
    """

    A,_ = vander1d(nodalPoints, degree)
    nf, nfx = vander1d(evaluationPoints, degree)
    shape = onp.linalg.solve(A.T, nf.T)
    dshape = onp.linalg.solve(A.T, nfx.T)
    return ShapeFunctions(shape, dshape) 


def vander1d(x, degree):
    x = onp.asarray(x)
    A = onp.zeros((x.shape[0], degree + 1))
    dA = onp.zeros((x.shape[0], degree + 1))
    domain = [0.0, 1.0]
    for i in range(degree + 1):
        p = onp.polynomial.Legendre.basis(i, domain=domain) 
        p *= onp.sqrt(2.0*i + 1.0) # keep polynomial orthonormal
        A[:, i] = p(x)
        dp = p.deriv()
        dA[:, i] = dp(x)
    return A, dA


def make_parent_element_2d(degree):
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

    lobattoPoints = get_lobatto_nodes_1d(degree)
    nPoints = int((degree + 1)*(degree + 2)/2)
    points = onp.zeros((nPoints, 2))
    point = 0
    for i in range(degree):
        for j in range(degree + 1 - i):
            k = degree - i - j
            points[point, 0] = (1.0 + 2.0*lobattoPoints[k] - lobattoPoints[j] - lobattoPoints[i])/3.0
            points[point, 1] = (1.0 + 2.0*lobattoPoints[j] - lobattoPoints[i] - lobattoPoints[k])/3.0
            point += 1
    points = np.asarray(points)

    vertexPoints = np.array([0, degree, nPoints - 1], dtype=np.int32)

    ii = onp.arange(degree + 1)
    jj = onp.cumsum(onp.flip(ii)) + ii
    kk = onp.flip(jj) - ii
    facePoints = np.array((ii,jj,kk), dtype=np.int32)

    interiorPoints = [i for i in range(nPoints) if i not in facePoints.ravel()]
    interiorPoints = np.array(interiorPoints, dtype=np.int32)
    
    return ParentElement(TRIANGLE_ELEMENT, int(degree), points, vertexPoints, facePoints, interiorPoints)


def pascal_triangle_monomials(degree):
    p = []
    q = []
    for i in range(1, degree + 2):
        monomialIndices = list(range(i))
        p += monomialIndices
        monomialIndices.reverse()
        q += monomialIndices
    return onp.column_stack((q,p))


def vander2d(x, degree):
    x = onp.asarray(x)
    nNodes = (degree+1)*(degree+2)//2
    pq = pascal_triangle_monomials(degree)

    # It's easier to process if the input arrays
    # always have the same shape
    # If a 1D array is given (a single point),
    # convert to the equivalent 2D array
    x = x.reshape(-1,2)
    
    # switch to bi-unit triangle (-1,-1)--(1,-1)--(-1,1)
    z = 2.0*x - 1.0
    
    def map_from_tri_to_square(xi):
        small = 1e-12
        # The mapping has a singularity at the vertex (-1, 1).
        # Handle that point specially.
        indexSingular = xi[:, 1] > 1.0 - small
        xiShifted = xi.copy()
        xiShifted[indexSingular, 1] = 1.0 - small
        eta = onp.zeros_like(xi)
        eta[:, 0] = 2.0*(1.0 + xiShifted[:, 0])/(1.0 - xiShifted[:, 1]) - 1.0
        eta[:, 1] = xiShifted[:, 1]
        eta[indexSingular, 0] = -1.0
        eta[indexSingular, 1] = 1.0
        
        # Jacobian of map. 
        # Actually, deta is just the first row of the Jacobian.
        # The second row is trivially [0, 1], so we don't compute it.
        # We just use that fact directly in the derivative Vandermonde
        # expressions.
        deta = onp.zeros_like(xi)
        deta[:, 0] = 2/(1 - xiShifted[:, 1])
        deta[:, 1] = 2*(1 + xiShifted[:, 0])/(1 - xiShifted[:, 1])**2
        return eta, deta
    
    E, dE = map_from_tri_to_square(onp.asarray(z))
    
    A = onp.zeros((x.shape[0], nNodes))
    Ax = A.copy()
    Ay = A.copy()
    N1D = onp.polynomial.Polynomial([0.5, -0.5])
    for i in range(nNodes):
        p = onp.polynomial.Legendre.basis(pq[i, 0])
        
        # SciPy's polynomials use the deprecated poly1d type
        # of NumPy. To convert to the modern Polynomial type,
        # we need to reverse the order of the coefficients.
        qPoly1d = special.jacobi(pq[i, 1], 2*pq[i, 0] + 1, 0)
        q = onp.polynomial.Polynomial(qPoly1d.coef[::-1])
        
        for j in range(pq[i, 0]):
            q *= N1D
        
        # orthonormality weight
        weight = onp.sqrt((2*pq[i,0] + 1) * 2*(pq[i, 0] + pq[i, 1] + 1))
        
        A[:, i] = weight*p(E[:, 0])*q(E[:, 1])
        
        # derivatives
        dp = p.deriv()
        dq = q.deriv()
        Ax[:, i] = 2*weight*dp(E[:, 0])*q(E[:, 1])*dE[:, 0]
        Ay[:, i] = 2*weight*(dp(E[:, 0])*q(E[:, 1])*dE[:, 1]
                             + p(E[:, 0])*dq(E[:, 1]))
        
    return A, Ax, Ay


def shape2d(degree, nodalPoints, evaluationPoints):
    A, _, _ = vander2d(nodalPoints, degree)
    nf, nfx, nfy = vander2d(evaluationPoints, degree)
    shapes = onp.linalg.solve(A.T, nf.T).T
    dshapes = onp.zeros(shapes.shape + (2,)) # shape is (nQuadPoints, nNodes, 2)
    dshapes[:, :, 0] = onp.linalg.solve(A.T, nfx.T).T
    dshapes[:, :, 1] = onp.linalg.solve(A.T, nfy.T).T
    return ShapeFunctions(np.asarray(shapes), np.asarray(dshapes))


def compute_shapes(parentElement, evaluationPoints):
    if parentElement.elementType == LINE_ELEMENT:
        return shape1d(parentElement.degree, parentElement.coordinates, evaluationPoints)
    elif parentElement.elementType == TRIANGLE_ELEMENT:
        return shape2d(parentElement.degree, parentElement.coordinates, evaluationPoints)
    elif parentElement.elementType == TRIANGLE_ELEMENT_WITH_BUBBLE:
        return shape2dBubble(parentElement, evaluationPoints)
    else:
        raise ValueError('Unknown element type.')


def make_parent_element_2d_with_bubble(degree):
    baseMaster = make_parent_element_2d(degree)
    bubbleMaster = make_parent_element_2d(degree + 1)

    nNodesFromBase = num_nodes(baseMaster) - baseMaster.interiorNodes.size
    nBubbleNodes = bubbleMaster.interiorNodes.shape[0]
    nNodes = nNodesFromBase + nBubbleNodes

    retainedBaseNodes = np.full(num_nodes(baseMaster), True)
    retainedBaseNodes = retainedBaseNodes.at[baseMaster.interiorNodes].set(False)

    coords = np.zeros((nNodes, 2))
    coords = coords.at[:nNodesFromBase].set(baseMaster.coordinates[retainedBaseNodes])
    coords = coords.at[nNodesFromBase:].set(bubbleMaster.coordinates[bubbleMaster.interiorNodes])

    vertexNodes = np.array([0, degree , nNodesFromBase - 1], dtype=np.int32)

    ii = onp.arange(degree + 1)
    jj = onp.array([i for i in range(degree, 3*degree, 2)] + [nNodesFromBase - 1])
    kk = onp.array([i for i in reversed(range(degree + 1, nNodesFromBase, 2))] + [0])
    faceNodes = np.array((ii,jj,kk), dtype=np.int32)

    interiorNodes =  np.arange(nNodesFromBase, nNodesFromBase + nBubbleNodes, dtype=np.int32)

    return ParentElement(TRIANGLE_ELEMENT_WITH_BUBBLE, degree, coords, vertexNodes,
                         faceNodes, interiorNodes)


def shape2dBubble(refElement, evaluationPoints):
    # base shape function values at eval points
    baseElement = make_parent_element_2d(refElement.degree)
    baseShapes, baseShapeGrads = shape2d(baseElement.degree, baseElement.coordinates, evaluationPoints)
    nodesFromBase = np.setdiff1d(np.arange(num_nodes(baseElement)),
                                 baseElement.interiorNodes,
                                 assume_unique=True)
    baseShapes = baseShapes[:, nodesFromBase]
    baseShapeGrads = baseShapeGrads[:, nodesFromBase, :]

    # base shape functions at bubble nodes
    bubbleElement = make_parent_element_2d(refElement.degree + 1)
    baseShapesAtBubbleNodes, _ = shape2d(baseElement.degree, baseElement.coordinates, bubbleElement.coordinates[bubbleElement.interiorNodes])
    baseShapesAtBubbleNodes = baseShapesAtBubbleNodes[:, nodesFromBase]

    # bubble function values at eval points
    bubbleShapes, bubbleShapeGrads = shape2d(bubbleElement.degree, bubbleElement.coordinates, evaluationPoints)
    bubbleShapes = bubbleShapes[:, bubbleElement.interiorNodes]
    bubbleShapeGrads = bubbleShapeGrads[:, bubbleElement.interiorNodes, :]

    baseShapes = baseShapes - bubbleShapes@baseShapesAtBubbleNodes
    shapes = np.hstack((baseShapes, bubbleShapes))

    baseShapeGrads = baseShapeGrads - np.einsum('qai,ab->qbi', bubbleShapeGrads, baseShapesAtBubbleNodes)
    dshapes = np.hstack((baseShapeGrads, bubbleShapeGrads))

    return ShapeFunctions(shapes, dshapes)


def num_nodes(master):
    return master.coordinates.shape[0]
