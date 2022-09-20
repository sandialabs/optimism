from collections import namedtuple
import numpy as onp
from scipy import special


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
    
    p = onp.polynomial.Legendre.basis(degree, domain=[0.0, 1.0])
    dp = p.deriv()
    xInterior = dp.roots()
    xn = onp.hstack(([0.0], xInterior, [1.0]))
    vertexPoints = onp.array([0, degree], dtype=onp.int32)
    interiorPoints = onp.arange(1, degree, dtype=onp.int32)
    return MasterElement(int(degree), xn, vertexPoints, None, interiorPoints)


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

    lobattoPoints = make_master_line_element(degree).coordinates
    nPoints = int((degree + 1)*(degree + 2)/2)
    points = onp.zeros((nPoints, 2))
    point = 0
    for i in range(degree):
        for j in range(degree + 1 - i):
            k = degree - i - j
            points[point, 0] = (1.0 + 2.0*lobattoPoints[k] - lobattoPoints[j] - lobattoPoints[i])/3.0
            points[point, 1] = (1.0 + 2.0*lobattoPoints[j] - lobattoPoints[i] - lobattoPoints[k])/3.0
            point += 1

    vertexPoints = onp.array([0, degree, nPoints - 1], dtype=onp.int32)

    ii = onp.arange(degree + 1)
    jj = onp.cumsum(onp.flip(ii)) + ii
    kk = onp.flip(jj) - ii
    facePoints = onp.array((ii,jj,kk), dtype=onp.int32)

    interiorPoints = [i for i in range(nPoints) if i not in facePoints.ravel()]
    interiorPoints = onp.array(interiorPoints, dtype=onp.int32)
    
    return MasterElement(int(degree), points, vertexPoints, facePoints, interiorPoints)


# deprecate
# use vander1d
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

def vander1d(x, degree):
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
        

def shape1d(master1d, evaluationPoints):
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
    
    A,_ = vander1d(master1d.coordinates, master1d.degree)
    nf, nfx = vander1d(evaluationPoints, master1d.degree)
    return onp.linalg.solve(A.T, nf.T), onp.linalg.solve(A.T, nfx.T)
    

def compute_1D_shape_function_values(nodalPoints, evaluationPoints, degree):
    """Evalute 1D shape functions at points in the master element.
    
    shapes: [nNodes, nEvalPoints]
    """
    A = make_vandermonde_1D(nodalPoints, degree)
    nf = make_vandermonde_1D(evaluationPoints, degree)
    print("A=\n", A)
    print("nf=\n", nf)
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
        monomialIndices = list(range(i))
        p += monomialIndices
        monomialIndices.reverse()
        q += monomialIndices
    return onp.column_stack((q,p))


def compute_vandermonde_tri(x, degree):
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

def vander2d(x, degree):
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
        Ax[:, i] = weight*dp(E[:, 0])*q(E[:, 1])*dE[:, 0]
        Ay[:, i] = weight*(dp(E[:, 0])*q(E[:, 1])*dE[:, 1] + 
                           p(E[:, 0])*dq(E[:, 1]))
        
    return A, Ax, Ay

def shape2d(masterElement, evaluationPoints):
    A, _, _ = vander2d(masterElement.coordinates, masterElement.degree)
    nf, nfx, nfy = vander2d(evaluationPoints, masterElement.degree)
    shapes = onp.linalg.solve(A.T, nf.T).T
    dshapes = onp.zeros(shapes.shape + (2,))
    dshapes[:, :, 0] = onp.linalg.solve(A.T, nfx.T).T
    dshapes[:, :, 1] = onp.linalg.solve(A.T, nfy.T).T
    return shapes, dshapes

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
