from optimism.JaxConfig import *
from jax.numpy.linalg import norm, eigh


def energy(A, b, s):
    return 0.5*s@(A@s) + s@b


def pnorm_squared(bvv, sig):
    return bvv@(1.0/(sig*sig))


def qnorm_squared(bvv, sig):
    return bvv@(1.0/(sig*sig*sig))


def solve(A, b, Delta):
    sig, v = eigh(A)
    bv = v.T@b
    bvv = bv*bv

    # Check if solution is inside the trust region
    if sig[0]>0 and norm(bv/sig) < Delta:
        return -v@(bv/sig)

    # if we get here, the solution must be on the tr boundary 
    
    sigScale = np.mean( np.abs(sig) )
    eps = 1e-12 * sigScale
    minSig = sig[0]

    # consider bounding the initial guess, see More' Sorenson paper
    lam = -minSig + eps if minSig < eps else 0.0

    #try to solve this for lam:
    #(A + lam I)p = -b, such that norm(p) = Delta
    
    # Check for the hard case
    if minSig < eps and norm(bv/(sig+lam)) < Delta:
        p = -v@(bv/(sig+lam))
        z = v[0]
        pz = p@z
        pp = p@p
        ddmpp = Delta*Delta-pp
        tau = ddmpp / (pz + np.sign(pz)*np.sqrt(pz*pz + ddmpp))
        return p + tau * z

    pNormSq = pnorm_squared(bvv, sig+lam)
    pNorm = np.sqrt(pNormSq)
    bError = (pNorm - Delta)/Delta
    #print('\nberror = ', bError)

    # consider an out if it doesnt converge, or use a better initial guess, or bound the lam from below and above.
    while np.abs(bError) > 1e-9:
        qNormSq = qnorm_squared(bvv, sig+lam)
        lam += (pNormSq / qNormSq) * bError
        pNormSq = pnorm_squared(bvv, sig+lam)
        pNorm = np.sqrt(pNormSq)
        bError = (pNorm - Delta)/Delta
        #print('\nberror = ', bError)

    return -v@(bv/(sig+lam))


