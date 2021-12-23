from optimism.JaxConfig import *
from optimism import Surface
from optimism.contact import SmoothMinMax

def norm_squared(vec):
    return vec[0]*vec[0] + vec[1]*vec[1]


def dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]


def cross(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]


def cpp_line(edge, p):
    a = edge[0]
    b = edge[1]
    v = b-a
    t = -dot(v,a-p) / norm_squared(v)
    return (1.0-t)*a + t*b, t


def cpp(edge, p):
    a = edge[0]
    b = edge[1]
    v = b-a
    t = -dot(v,a-p) / norm_squared(v)
    t = np.where(t < 0., 0.0, t)
    t = np.where(t > 1., 1.0, t)
    return (1.0-t)*a + t*b, t


def cpp_distance(edge, p):
    norm = Surface.compute_normal(edge)
    cppPoint,t = cpp_line(edge, p)
    dist = dot(norm, p-cppPoint)
    sgn = np.sign(dist)
    sgn = np.where(sgn==0, 1.0, sgn)
    dist = np.where(t < 0., np.sqrt(norm_squared(edge[0]-p)) * sgn, dist)
    dist = np.where(t > 1., np.sqrt(norm_squared(edge[1]-p)) * sgn, dist)
    return dist


#unused, but a nice function anyways
def smoothstep(x):
    x = np.minimum(x, 1.0)
    x = np.maximum(x, 0.0)
    return 3*x**2 - 2*x**3 


def area(p0,p1,p2):
    return 0.5*(p0[0]*(p1[1]-p2[1]) + p1[0]*(p2[1]-p0[1] ) + p2[0]*(p0[1]-p1[1]) )


def smooth_distance(twoEdges, p, smoothingTol):
    # this can probably be computed externally for efficiency
    a1 = area(twoEdges[0][0], twoEdges[0][1], twoEdges[1][0])
    a2 = area(twoEdges[1][0], twoEdges[1][1], twoEdges[0][0])

    sign = -np.sign(a1+a2)
    sign = np.where(sign==0, 1.0, sign)
    
    p0 = cpp(twoEdges[0], p)[0]
    p1 = cpp(twoEdges[1], p)[0]
        
    d0 = p-p0
    d1 = p-p1
    
    n0 = Surface.compute_normal(twoEdges[0])
    n1 = Surface.compute_normal(twoEdges[1])

    pd0 = dot(d0, n0)
    pd1 = dot(d1, n1)

    # tolerance approaches 0 as the normal point in the same direction
    # this ensures the min smoothing gap goes to zero as the outer boundary flips between
    # concave and convex

    crossN = np.abs(cross(n0, n1))
    tol = np.where(crossN > 1e-14, crossN*smoothingTol, 0.0)
    #sign = 1.0
    return sign*SmoothMinMax.safe_min(sign*pd0, sign*pd1, tol)


    
