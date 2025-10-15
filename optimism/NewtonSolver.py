import numpy as onp
from scipy.sparse.linalg import LinearOperator, gmres

from optimism.JaxConfig import *


Settings = namedtuple('Settings', ['relative_gmres_tol',
                                   'max_gmres_iters'])

def construct_quadratic(ps):
    c = ps[0]
    b = ps[2]
    a = (ps[1] - ps[0] - ps[2])
    return lambda s: a*s*s + b*s + c


def compute_min_p(ps, bounds):
    c = ps[0]
    b = ps[2]
    a = (ps[1] - ps[0] - ps[2])
    
    if a <= 0:
        return bounds[0] if ps[0] < ps[1] else bounds[1]
    else:
        p = construct_quadratic(ps)
        quadMin = -b/(2*a)
        return min(max(quadMin, bounds[0]), bounds[1])


def make_scipy_linear_function(linear_function):
    def linear_op(v):
        # The v is going into a jax function (probably a jvp).
        # Sometimes scipy passes in an array of dtype int, which breaks
        # jax tracing and differentiation, so explicitly set type to
        # something jax can handle.
        jax_v = np.array(v, dtype=np.float64)
        jax_Av = linear_function(jax_v)
        # The result is going back into a scipy solver, so convert back
        # to a standard numpy array.
        return onp.array(jax_Av)
    return linear_op


def newton_step(residual, residual_jvp, x, settings=Settings(1e-2,100), precond=None):
    sz = x.size
    A = LinearOperator((sz, sz), make_scipy_linear_function(residual_jvp))
    r = onp.array(residual(x))

    numIters = 0
    def callback(xk):
        nonlocal numIters
        numIters += 1

    relTol = settings.relative_gmres_tol
    maxIters = settings.max_gmres_iters
    
    if precond is not None:
        M = LinearOperator((sz, sz), make_scipy_linear_function(precond))
    else:
        M = None
        
    dx, exitcode = gmres(A, r, rtol=relTol, atol=0, M=M, callback_type='legacy', callback=callback, maxiter=maxIters)
    print('Number of GMRES iters = ', numIters)
        
    return -dx, exitcode


def globalized_newton_step(residual, linear_op, x, etak=1e-3, t=1e-4, maxLinesearchIters=4, precond=None):
    r0 = residual(x)
    rEnergy0 = 0.5*np.linalg.norm(r0)**2
    
    s, exitcode = newton_step(residual, linear_op, x, precond=precond)

    if exitcode != 0:
        print('Failed to solve linearized newton step with gmres')
        return 0.0
    
    rN = residual(x + s)
    rEnergyN = 0.5*np.linalg.norm(rN)**2
    
    for count in range(maxLinesearchIters):

        print('linesearch iter = ', count)
        
        minImprove = 1.0-t*(1.0-etak)
        if rEnergyN < minImprove*rEnergy0:
            return s

        def energy(t):
            r = residual(x+t*s)
            return 0.5*r@r
        
        drEnergy0 = grad(energy)(0.0)

        if drEnergy0 >= 0.0: return 0.0
        
        theta = compute_min_p([rEnergy0, rEnergyN, drEnergy0], [0.01, 0.5])
        
        s *= theta
        etak = 1.0-theta*(1.0-etak)
        
        rN = residual(x + s)
        rEnergyN = 0.5*np.linalg.norm(rN)**2

        #raise NameError("Exceeded the maximum number of linesearch cutback tried.")
        
    return 0.0
