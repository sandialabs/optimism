from optimism.JaxConfig import *


def cross2(vec1, vec2):
    return vec1[0]*vec2[1]-vec1[1]*vec2[0]


def dot2(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]


#@jit
def compute_ray_trace_distance_and_location(edge, ray):
    tol = 1e-13
    r = edge[1]-edge[0]
    s = ray[1]
    p = edge[0]
    q = ray[0]
    rCrossS = cross2(r, s)
    absrCrossS = np.abs(rCrossS)
    absrCrossS = np.maximum(tol*tol*np.abs(dot2(r, s)), absrCrossS)
    rCrossS = np.copysign(absrCrossS, rCrossS)
    
    qmp = q-p
    t = cross2(qmp, s) / rCrossS
    u = cross2(qmp, r) / rCrossS
    return u, t


#@jit
def compute_valid_ray_trace_distance(edge, ray):
    u,t = compute_ray_trace_distance_and_location(edge, ray)
    tInvalid = np.maximum(np.maximum(-t, t-1.), 0.)
    u = if_then_else(tInvalid > 0, np.inf, u)
    return u, t


#@jit
def compute_smoothing_function(u, t, eps):
    dx = if_then_else(t>1.0, t-1.0, -t)
    isOutside = dx>0.0
    return if_then_else(isOutside, u + 0.5*dx*dx, u)
    #return if_then_else(isOutside, u + 0.5*dx*dx, u)

    #isWayOutside = dx >= eps
    #dx = if_then_else(isWayOutside, 1.0, dx)
    #smooth1 = if_then_else(dx > 0, eps/(eps-dx) - dx/eps, 1.0)
    #return if_then_else(isWayOutside, np.inf, smooth1)


#@jit
def compute_valid_ray_trace_distance_smoothed(edge, ray, eps):
    u,t = compute_ray_trace_distance_and_location(edge, ray)    
    smoothedU = compute_smoothing_function(u, t, eps)
    return smoothedU, t

