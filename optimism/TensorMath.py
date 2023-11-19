"""Provide differentiable operations on 3x3 tensors."""

from functools import partial
import jax
import jax.numpy as np

from optimism.JaxConfig import if_then_else
from optimism import Math

def trace(A):
    return A[0, 0] + A[1, 1] + A[2, 2]

def det(A):
    return A[0, 0]*A[1, 1]*A[2, 2] + A[0, 1]*A[1, 2]*A[2, 0] + A[0, 2]*A[1, 0]*A[2, 1] \
        - A[0, 0]*A[1, 2]*A[2, 1] - A[0, 1]*A[1, 0]*A[2, 2] - A[0, 2]*A[1, 1]*A[2, 0]

def inv(A):
    invA00 = A[1, 1]*A[2, 2] - A[1, 2]*A[2, 1]
    invA01 = A[0, 2]*A[2, 1] - A[0, 1]*A[2, 2]
    invA02 = A[0, 1]*A[1, 2] - A[0, 2]*A[1, 1]
    invA10 = A[1, 2]*A[2, 0] - A[1, 0]*A[2, 2]
    invA11 = A[0, 0]*A[2, 2] - A[0, 2]*A[2, 0]
    invA12 = A[0, 2]*A[1, 0] - A[0, 0]*A[1, 2]
    invA20 = A[1, 0]*A[2, 1] - A[1, 1]*A[2, 0]
    invA21 = A[0, 1]*A[2, 0] - A[0, 0]*A[2, 1]
    invA22 = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    invA = (1.0/det(A)) * np.array([[invA00, invA01, invA02],
                                    [invA10, invA11, invA12],
                                    [invA20, invA21, invA22]])
    return invA

def compute_deviatoric_tensor(A):
    dil = trace(A)
    return A - (dil/3.)*np.identity(3)

def dev(strain): return compute_deviatoric_tensor(strain)

def sym(A):
    return 0.5*(A + A.T)

def skw(A):
    return 0.5*(A - A.T)

def norm(A):
    return Math.safe_sqrt(A.ravel() @ A.ravel())

def norm_of_deviator_squared(tensor):
    dev = compute_deviatoric_tensor(tensor)
    return np.tensordot(dev,dev)

def norm_of_deviator(tensor):
    return norm( compute_deviatoric_tensor(tensor) )

def mises_invariant(stress):
    return np.sqrt(1.5)*norm_of_deviator(stress)

def triaxiality(A):
    mean_normal = trace(A)/3.0
    mises_norm = mises_invariant(A)
    # avoid division by zero in case of spherical tensor
    mises_norm += np.finfo(np.dtype("float64")).eps
    return mean_normal/mises_norm

def tensor_2D_to_3D(H):
    return np.zeros((3,3)).at[ 0:H.shape[0], 0:H.shape[1] ].set(H)

def gradient_2D_to_axisymmetric(dudX_2D, u, X):
    dudX = tensor_2D_to_3D(dudX_2D)
    dudX = dudX.at[2, 2].set(u[0]/X[0])
    return dudX

def eigen_sym33_non_unit(tensor):
    """Compute eigen values and vectors of a symmetric 3x3 tensor.

    Note, returned eigen vectors may not be unit length
    Note, this routine involves high powers of the input tensor (~M^8).
    Thus results can start to denormalize when the infinity norm of the input
    tensor falls outside the range 1.0e-40 to 1.0e+40.
    Outside this range use eigen_sym33_unit
    """
    cxx = tensor[0,0]
    cyy = tensor[1,1]
    czz = tensor[2,2]
    cxy = 0.5*(tensor[0,1]+tensor[1,0])
    cyz = 0.5*(tensor[1,2]+tensor[2,1])
    czx = 0.5*(tensor[2,0]+tensor[0,2])

    c1 = (cxx + cyy + czz)/(3.0)

    cxx -= c1
    cyy -= c1
    czz -= c1
  
    cxy_cxy = cxy*cxy
    cyz_cyz = cyz*cyz
    czx_czx = czx*czx
    cxx_cyy = cxx*cyy
    
    c2 = cxx_cyy + cyy*czz + czz*cxx - cxy_cxy - cyz_cyz - czx_czx
    

    c2Negative = c2 < 0
    denom = np.where(c2Negative, c2, 1.0)
    ThreeOverA = np.where(c2Negative, -3.0/denom, 1.0)
    sqrtThreeOverA = np.where(c2Negative, np.sqrt(ThreeOverA), 1.0)
    
    c3 = cxx*cyz_cyz + cyy*czx_czx - 2.0*cxy*cyz*czx + czz*(cxy_cxy - cxx_cyy)

    rr = -0.5*c3*ThreeOverA*sqrtThreeOverA
    
    arg = np.minimum(abs(rr), 1.0) # Check in the case rr = -1-eps
    
    cos_thd3 = cos_of_acos_divided_by_3(arg)

    two_cos_thd3 = 2.0*cos_thd3*np.sign(rr)

    eval2 = np.where(c2Negative, two_cos_thd3/sqrtThreeOverA, 1.0)
    
    crow0 = np.array([cxx - eval2, cxy,         czx        ])
    crow1 = np.array([cxy,         cyy - eval2, cyz        ])
    crow2 = np.array([czx,         cyz,         czz - eval2])

    #
    # do QR decomposition with column pivoting
    #
    k0 = crow0[0]*crow0[0] + cxy_cxy           + czx_czx
    k1 = cxy_cxy           + crow1[1]*crow1[1] + cyz_cyz
    k2 = czx_czx           + cyz_cyz           + crow2[2]*crow2[2]
    
    # returns zero or nan
    k0gk1 = k1<=k0
    k0gk2 = k2<=k0
    k1gk2 = k2<=k1
    
    k0_largest = k0gk1 & k0gk2
    k1_largest = k1gk2 & (~ k0gk1)
    k2_largest = ~ (k0_largest | k1_largest)
    k_largest = np.array([k0_largest, k1_largest, k2_largest])

    k_row1_0 = if_then_else(k0_largest, crow0[0], 0.0)   \
        +         if_then_else(k1_largest, crow1[0], 0.0) \
        +         if_then_else(k2_largest, crow2[0], 0.0)

    k_row1_1 = if_then_else(k0_largest, crow0[1], 0.0)   \
        +         if_then_else(k1_largest, crow1[1], 0.0) \
        +         if_then_else(k2_largest, crow2[1], 0.0)

    k_row1_2 = if_then_else(k0_largest, crow0[2], 0.0)   \
        +         if_then_else(k1_largest, crow1[2], 0.0) \
        +         if_then_else(k2_largest, crow2[2], 0.0)

    k_row1 = np.array([k_row1_0, k_row1_1, k_row1_2])
    
    row2_0 = if_then_else(k0_largest, crow1[0], crow0[0])
    row2_1 = if_then_else(k0_largest, crow1[1], crow0[1])
    row2_2 = if_then_else(k0_largest, crow1[2], crow0[2])
    row2 = np.array([row2_0, row2_1, row2_2])

    row3_0 = if_then_else(k2_largest, crow1[0], crow2[0])
    row3_1 = if_then_else(k2_largest, crow1[1], crow2[1])
    row3_2 = if_then_else(k2_largest, crow1[2], crow2[2])
    row3 = np.array([row3_0, row3_1, row3_2])

    ki_ki = 1.0 / ( if_then_else(k0_largest, k0, 0.0)   \
                    + if_then_else(k1_largest, k1, 0.0) \
                    + if_then_else(k2_largest, k2, 0.0) )
    
    ki_dpr1 = ki_ki*(k_row1[0]*row2[0] + k_row1[1]*row2[1] + k_row1[2]*row2[2])
    ki_dpr2 = ki_ki*(k_row1[0]*row3[0] + k_row1[1]*row3[1] + k_row1[2]*row3[2])

    row2 = row2 - ki_dpr1*k_row1
    row3 = row3 - ki_dpr2*k_row1

    a0 = row2[0]*row2[0] + row2[1]*row2[1] + row2[2]*row2[2]
    a1 = row3[0]*row3[0] + row3[1]*row3[1] + row3[2]*row3[2]

    a0lea1 = a0 <= a1

    a_row2 = if_then_else(a0lea1, row3, row2)
    ai_ai = 1.0 / if_then_else(a0lea1, a1, a0)
    
    evec2 = np.array([k_row1[1]*a_row2[2] - k_row1[2]*a_row2[1],
                      k_row1[2]*a_row2[0] - k_row1[0]*a_row2[2],
                      k_row1[0]*a_row2[1] - k_row1[1]*a_row2[0]])

    k_atr11 = cxx*k_row1[0] + cxy*k_row1[1] + czx*k_row1[2]
    k_atr21 = cxy*k_row1[0] + cyy*k_row1[1] + cyz*k_row1[2]
    k_atr31 = czx*k_row1[0] + cyz*k_row1[1] + czz*k_row1[2]

    a_atr12 = cxx*a_row2[0] + cxy*a_row2[1] + czx*a_row2[2]
    a_atr22 = cxy*a_row2[0] + cyy*a_row2[1] + cyz*a_row2[2]
    a_atr32 = czx*a_row2[0] + cyz*a_row2[1] + czz*a_row2[2]

    rm2xx     = (k_row1[0]*k_atr11 + k_row1[1]*k_atr21 + k_row1[2]*k_atr31)*ki_ki
    k_a_rm2xy = (k_row1[0]*a_atr12 + k_row1[1]*a_atr22 + k_row1[2]*a_atr32)
    rm2yy     = (a_row2[0]*a_atr12 + a_row2[1]*a_atr22 + a_row2[2]*a_atr32)*ai_ai
    rm2xy_rm2xy = k_a_rm2xy*k_a_rm2xy*ai_ai*ki_ki

    #
    # Wilkinson shift
    #
    b = 0.5*(rm2xx-rm2yy)

    sqrtTerm = Math.safe_sqrt(b*b+rm2xy_rm2xy)*np.sign(b)
    #sqrtTerm = np.sqrt(b*b+rm2xy_rm2xy)*np.sign(b)
    
    eval0 = rm2yy + b - sqrtTerm
    eval1 = rm2xx + rm2yy - eval0

    rm2xx -= eval0
    rm2yy -= eval0

    rm2xx2 = rm2xx*rm2xx
    rm2yy2 = rm2yy*rm2yy

    fac1 = if_then_else(rm2xx2 < rm2yy2, k_a_rm2xy*ai_ai, rm2xx)
    fac2 = if_then_else(rm2xx2 < rm2yy2, rm2yy, ki_ki*k_a_rm2xy)

    evec0 = fac1*a_row2 - fac2*k_row1

    rm2xx2iszero = rm2xx2 == (0.0)
    rm2xy_rm2xyiszero = rm2xy_rm2xy == (0.0)
    both_zero = rm2xx2iszero & rm2xy_rm2xyiszero

    # check degeneracy
    
    evec0 = if_then_else(both_zero, a_row2, evec0)

    evec1 = np.array([evec2[1]*evec0[2] - evec2[2]*evec0[1],
                      evec2[2]*evec0[0] - evec2[0]*evec0[2],
                      evec2[0]*evec0[1] - evec2[1]*evec0[0]])

    eval0 = eval0 + c1
    eval1 = eval1 + c1
    eval2 = eval2 + c1
    
    c2tol = (c1*c1)*(-1.0e-30)

    c2lsmall_neg = c2 < c2tol
    
    eval0 = if_then_else(c2lsmall_neg, eval0, c1)
    eval1 = if_then_else(c2lsmall_neg, eval1, c1)
    eval2 = if_then_else(c2lsmall_neg, eval2, c1)

    evec0 = if_then_else(c2lsmall_neg, evec0, np.array([1.0, 0.0, 0.0]))
    evec1 = if_then_else(c2lsmall_neg, evec1, np.array([0.0, 1.0, 0.0]))
    evec2 = if_then_else(c2lsmall_neg, evec2, np.array([0.0, 0.0, 1.0]))
    
    evals = np.array([eval0, eval1, eval2])
    evecs = np.column_stack((evec0,evec1,evec2))

    #idx = np.arange(3)  # np.argsort(evals)
    idx = np.argsort(evals)
    
    return evals[idx],evecs[:,idx]


def eigen_sym33_unit(tensor):
    cmax = np.linalg.norm(tensor, ord=np.inf)
    cmaxInv = if_then_else(cmax > 0.0, 1.0/cmax, 1.0)
    scaledTensor = cmaxInv * tensor
   
    evals, evecs = eigen_sym33_non_unit(scaledTensor)
    
    evec0 = evecs[:,0]/np.linalg.norm(evecs[:,0])
    evec1 = evecs[:,1]/np.linalg.norm(evecs[:,1])
    evec2 = evecs[:,2]/np.linalg.norm(evecs[:,2])
    
    evecs = np.column_stack((evec0,evec1,evec2))
    evals = cmax*evals

    return (evals,evecs)


# Helper function for 3x3 spectral decompositions
# Pade approximation to cos( acos(x)/3 )
# was obtained from Mathematica with the following commands:
#  
# Needs["FunctionApproximations`"]
# r1 = MiniMaxApproximation[Cos[ArcCos[x]/3], {x, {0, 1}, 6, 5}, WorkingPrecision -> 18, MaxIterations -> 500]
#
# 6 and 5 indicate the polynomial order in the numerator and denominator.
def cos_of_acos_divided_by_3(x):
    
    x2 = x*x;
    x4 = x2*x2;

    numer = 0.866025403784438713 + 2.12714890259493060 * x + \
        ( ( 1.89202064815951569  + 0.739603278343401613 * x ) * x2 + \
          ( 0.121973926953064794 + x * (0.00655637626263929360 + 0.0000390884982780803443 * x) ) *x4 )

    denom =     1.0 + 2.26376989330935617* x + \
        ( ( 1.80461009751278976 + 0.603976798217196003 * x ) * x2 + \
         ( 0.0783255761115461708 + 0.00268525944538021629 * x) * x4 )
    
    return numer/denom


@jax.custom_jvp
def mtk_log_sqrt(A):
    lam,V = eigen_sym33_unit(A)
    return V @ np.diag(0.5*np.log(lam)) @ V.T


@mtk_log_sqrt.defjvp
def mtk_log_sqrt_jvp(Cpack, Hpack):
    C, = Cpack
    H, = Hpack

    logSqrtC = mtk_log_sqrt(C)
    lam,V = eigen_sym33_unit(C)
    
    lam1 = lam[0]
    lam2 = lam[1]
    lam3 = lam[2]
    
    e1 = V[:,0]
    e2 = V[:,1]
    e3 = V[:,2]
    
    hHat = 0.5 * (V.T @ H @ V)

    l1111 = hHat[0,0] / lam1
    l2222 = hHat[1,1] / lam2
    l3333 = hHat[2,2] / lam3
    
    l1212 = 0.5*(hHat[0,1]+hHat[1,0]) * relative_log_difference(lam1, lam2)
    l2323 = 0.5*(hHat[1,2]+hHat[2,1]) * relative_log_difference(lam2, lam3)
    l3131 = 0.5*(hHat[2,0]+hHat[0,2]) * relative_log_difference(lam3, lam1)

    t00 = l1111 * e1[0] * e1[0] + l2222 * e2[0] * e2[0] + l3333 * e3[0] * e3[0] + \
        2 * l1212 * e1[0] * e2[0] + \
        2 * l2323 * e2[0] * e3[0] + \
        2 * l3131 * e3[0] * e1[0]
    t11 = l1111 * e1[1] * e1[1] + l2222 * e2[1] * e2[1] + l3333 * e3[1] * e3[1] + \
        2 * l1212 * e1[1] * e2[1] + \
        2 * l2323 * e2[1] * e3[1] + \
        2 * l3131 * e3[1] * e1[1]
    t22 = l1111 * e1[2] * e1[2] + l2222 * e2[2] * e2[2] + l3333 * e3[2] * e3[2] + \
        2 * l1212 * e1[2] * e2[2] + \
        2 * l2323 * e2[2] * e3[2] + \
        2 * l3131 * e3[2] * e1[2]

    t01 = l1111 * e1[0] * e1[1] + l2222 * e2[0] * e2[1] + l3333 * e3[0] * e3[1] + \
        l1212 * (e1[0] * e2[1] + e2[0] * e1[1]) + \
        l2323 * (e2[0] * e3[1] + e3[0] * e2[1]) + \
        l3131 * (e3[0] * e1[1] + e1[0] * e3[1])
    t12 = l1111 * e1[1] * e1[2] + l2222 * e2[1] * e2[2] + l3333 * e3[1] * e3[2] + \
        l1212 * (e1[1] * e2[2] + e2[1] * e1[2]) + \
        l2323 * (e2[1] * e3[2] + e3[1] * e2[2]) + \
        l3131 * (e3[1] * e1[2] + e1[1] * e3[2])
    t20 = l1111 * e1[2] * e1[0] + l2222 * e2[2] * e2[0] + l3333 * e3[2] * e3[0] + \
        l1212 * (e1[2] * e2[0] + e2[2] * e1[0]) + \
        l2323 * (e2[2] * e3[0] + e3[2] * e2[0]) + \
        l3131 * (e3[2] * e1[0] + e1[2] * e3[0])
    
    sol = np.array([ [t00, t01, t20],
                     [t01, t11, t12],
                     [t20, t12, t22] ])
        
    return logSqrtC, sol


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def mtk_pow(A,m):
    lam,V = eigen_sym33_unit(A)
    return V @ np.diag(np.power(lam,m)) @ V.T


@mtk_pow.defjvp
def mtk_pow_jvp(m, Cpack, Hpack):
    C, = Cpack
    H, = Hpack

    powC = mtk_pow(C,m)
    lam,V = eigen_sym33_unit(C)
    
    lam1 = lam[0]
    lam2 = lam[1]
    lam3 = lam[2]
    
    e1 = V[:,0]
    e2 = V[:,1]
    e3 = V[:,2]
    
    hHat = m * (V.T @ H @ V)

    l1111 = hHat[0,0] * np.power(lam1, m-1)
    l2222 = hHat[1,1] * np.power(lam2, m-1)
    l3333 = hHat[2,2] * np.power(lam3, m-1)
    
    l1212 = 0.5*(hHat[0,1]+hHat[1,0]) * relative_log_difference(lam1, lam2)
    l2323 = 0.5*(hHat[1,2]+hHat[2,1]) * relative_log_difference(lam2, lam3)
    l3131 = 0.5*(hHat[2,0]+hHat[0,2]) * relative_log_difference(lam3, lam1)

    t00 = l1111 * e1[0] * e1[0] + l2222 * e2[0] * e2[0] + l3333 * e3[0] * e3[0] + \
        2 * l1212 * e1[0] * e2[0] + \
        2 * l2323 * e2[0] * e3[0] + \
        2 * l3131 * e3[0] * e1[0]
    t11 = l1111 * e1[1] * e1[1] + l2222 * e2[1] * e2[1] + l3333 * e3[1] * e3[1] + \
        2 * l1212 * e1[1] * e2[1] + \
        2 * l2323 * e2[1] * e3[1] + \
        2 * l3131 * e3[1] * e1[1]
    t22 = l1111 * e1[2] * e1[2] + l2222 * e2[2] * e2[2] + l3333 * e3[2] * e3[2] + \
        2 * l1212 * e1[2] * e2[2] + \
        2 * l2323 * e2[2] * e3[2] + \
        2 * l3131 * e3[2] * e1[2]

    t01 = l1111 * e1[0] * e1[1] + l2222 * e2[0] * e2[1] + l3333 * e3[0] * e3[1] + \
        l1212 * (e1[0] * e2[1] + e2[0] * e1[1]) + \
        l2323 * (e2[0] * e3[1] + e3[0] * e2[1]) + \
        l3131 * (e3[0] * e1[1] + e1[0] * e3[1])
    t12 = l1111 * e1[1] * e1[2] + l2222 * e2[1] * e2[2] + l3333 * e3[1] * e3[2] + \
        l1212 * (e1[1] * e2[2] + e2[1] * e1[2]) + \
        l2323 * (e2[1] * e3[2] + e3[1] * e2[2]) + \
        l3131 * (e3[1] * e1[2] + e1[1] * e3[2])
    t20 = l1111 * e1[2] * e1[0] + l2222 * e2[2] * e2[0] + l3333 * e3[2] * e3[0] + \
        l1212 * (e1[2] * e2[0] + e2[2] * e1[0]) + \
        l2323 * (e2[2] * e3[0] + e3[2] * e2[0]) + \
        l3131 * (e3[2] * e1[0] + e1[2] * e3[0])
    
    sol = np.array([ [t00, t01, t20],
                     [t01, t11, t12],
                     [t20, t12, t22] ])
        
    return powC, sol


def relative_log_difference_taylor(lam1, lam2):
    # Compute a more accurate (mtk::log(lam1) - log(lam2)) / (lam1-lam2) as lam1 -> lam2
    third2 = 2.0 / 3.0
    fifth2 = 2.0 / 5.0
    seventh2 = 2.0 / 7.0
    ninth2 = 2.0 / 9.0
    
    frac = (lam1 - lam2) / (lam1 + lam2)
    frac2 = frac*frac
    frac4 = frac2*frac2
    
    # relative tolerance of 0.05 for this approx (with more terms its valid over larger range)
    return (2.0 + third2 * frac2 + fifth2 * frac4 + seventh2 * frac4 * frac2 + ninth2 * frac4 * frac4) / (lam1 + lam2)


def relative_log_difference_no_tolerance_check(lam1, lam2):
  return np.log(lam1 / lam2) / (lam1 - lam2)


def relative_log_difference(lam1, lam2):
    haveLargeDiff = np.abs(lam1 - lam2) > 0.05 * np.minimum(lam1, lam2)
    lamFake = np.where(haveLargeDiff, lam2, 2.0*lam2)
    return np.where(haveLargeDiff,
                    relative_log_difference_no_tolerance_check(lam1, lamFake),
                    relative_log_difference_taylor(lam1, lam2))

#
# We should consider deprecating the following functions and use the mtk ones exclusively
#

def logh(A):
    d,V = np.linalg.eigh(A)
    return logh_from_eigen(d,V)


def logh_from_eigen(eVals, eVecs):
    return eVecs@np.diag(np.log(eVals))@eVecs.T

# C must be symmetric!
@jax.custom_jvp
def log_sqrt(C):
    return 0.5*logh(C)

@log_sqrt.defjvp
def log_jvp(Cpack, Hpack):
    C, = Cpack
    H, = Hpack

    logSqrtC = log_sqrt(C)
    lam,V = np.linalg.eigh(C)
    
    lam1 = lam[0]
    lam2 = lam[1]
    lam3 = lam[2]
    
    e1 = V[:,0]
    e2 = V[:,1]
    e3 = V[:,2]
    
    hHat = 0.5 * (V.T @ H @ V)

    l1111 = hHat[0,0] / lam1
    l2222 = hHat[1,1] / lam2
    l3333 = hHat[2,2] / lam3
    
    l1212 = 0.5*(hHat[0,1]+hHat[1,0]) * relative_log_difference(lam1, lam2)
    l2323 = 0.5*(hHat[1,2]+hHat[2,1]) * relative_log_difference(lam2, lam3)
    l3131 = 0.5*(hHat[2,0]+hHat[0,2]) * relative_log_difference(lam3, lam1)

    t00 = l1111 * e1[0] * e1[0] + l2222 * e2[0] * e2[0] + l3333 * e3[0] * e3[0] + \
        2 * l1212 * e1[0] * e2[0] + \
        2 * l2323 * e2[0] * e3[0] + \
        2 * l3131 * e3[0] * e1[0]
    t11 = l1111 * e1[1] * e1[1] + l2222 * e2[1] * e2[1] + l3333 * e3[1] * e3[1] + \
        2 * l1212 * e1[1] * e2[1] + \
        2 * l2323 * e2[1] * e3[1] + \
        2 * l3131 * e3[1] * e1[1]
    t22 = l1111 * e1[2] * e1[2] + l2222 * e2[2] * e2[2] + l3333 * e3[2] * e3[2] + \
        2 * l1212 * e1[2] * e2[2] + \
        2 * l2323 * e2[2] * e3[2] + \
        2 * l3131 * e3[2] * e1[2]

    t01 = l1111 * e1[0] * e1[1] + l2222 * e2[0] * e2[1] + l3333 * e3[0] * e3[1] + \
        l1212 * (e1[0] * e2[1] + e2[0] * e1[1]) + \
        l2323 * (e2[0] * e3[1] + e3[0] * e2[1]) + \
        l3131 * (e3[0] * e1[1] + e1[0] * e3[1])
    t12 = l1111 * e1[1] * e1[2] + l2222 * e2[1] * e2[2] + l3333 * e3[1] * e3[2] + \
        l1212 * (e1[1] * e2[2] + e2[1] * e1[2]) + \
        l2323 * (e2[1] * e3[2] + e3[1] * e2[2]) + \
        l3131 * (e3[1] * e1[2] + e1[1] * e3[2])
    t20 = l1111 * e1[2] * e1[0] + l2222 * e2[2] * e2[0] + l3333 * e3[2] * e3[0] + \
        l1212 * (e1[2] * e2[0] + e2[2] * e1[0]) + \
        l2323 * (e2[2] * e3[0] + e3[2] * e2[0]) + \
        l3131 * (e3[2] * e1[0] + e1[2] * e3[0])
    
    sol = np.array([ [t00, t01, t20],
                     [t01, t11, t12],
                     [t20, t12, t22] ])
        
    return logSqrtC, sol


