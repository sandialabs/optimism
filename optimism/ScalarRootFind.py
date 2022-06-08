from optimism.JaxConfig import *

from jax.lax import while_loop, custom_root

SolutionInfo = namedtuple('SolutionInfo', ['converged', 'iterations', 'function_calls'])

Settings = namedtuple('Settings', ['max_iters', 'x_tol', 'r_tol'])


def get_settings(max_iters=50, x_tol=1e-13, r_tol=1e-13):
    return Settings(max_iters, x_tol, r_tol)


def find_root(f, x0, bracket, settings):
    """Find a root of a nonlinear scalar-valued equation.

    Uses Newton's method, safeguarded with bisection.
    See rtsafe(...) from Numerical Recipes.

    Parameters
    ==========
    f : callable
        Scalar function of which to find a root.
    x0 : real
        Initial guess for root. The value of x0 should be within the
        range defined by bracket. If not, the initial guess will be
        automatically moved to the midpoint of the bracket.
    bracket : list or tuple of 2 reals
        Upper and lower bounds for the root search.
    settings : A settings object from this module
        Sets algorithmic settings.

    Returns
    =======
    x : real
        Argument of f such that f(x) = 0 (within provided tolerances)
    """
    return custom_root(f, x0, lambda F, X0: rtsafe_(F, X0, bracket, settings),
                       lambda g, y: y/g(1.0))


def rtsafe_(f, x0, bracket, settings):
    # Find root of a scalar function
    # Newton's method, safeguarded with bisection
    # from Numerical Recipes

    max_iters = settings.max_iters
    x_tol = settings.x_tol
    r_tol = settings.r_tol

    f_and_fprime = value_and_grad(f)
    
    # check that root is bracketed
    fl = f(bracket[0])
    fh = f(bracket[1])
    functionCalls = 2

    #print('b0, b1 = ', bracket[0], bracket[1])
    #print('fs = ', fl, fh)

    # If root is not bracketed, send sentinel value (nan)
    x0 = np.where(fl*fh < 0,
                  x0,
                  np.nan)

    # ORIENT THE SEARCH SO THAT F(XL) < 0.
    xl, xh = lax.cond(fl < 0,
                      lambda _: (bracket[0], bracket[1]),
                      lambda _:(bracket[1], bracket[0]),
                      None)

    # INITIALIZE THE GUESS FOR THE ROOT, THE ''STEP SIZE
    # BEFORE LAST'', AND THE LAST STEP
    midpoint = 0.5*(xl + xh)
    x0 = if_then_else(x0 > xh,
                      midpoint,
                      x0)
    x0 = if_then_else(x0 < xl,
                      midpoint,
                      x0)
    dxOld = np.abs(bracket[1] - bracket[0])
    dx = dxOld

    F, DF = f_and_fprime(x0)
    functionCalls += 1

    def cond(carry):
        root, dx, dxOld, F, DF, xl, xh, converged, i = carry
        keepLooping = ~converged & (i < max_iters)
        return keepLooping

    def loop_body(carry):
        root, dx, dxOld, F, DF, xl, xh, converged, i = carry
        
        bisectStep = 0.5*(xh - xl)
        newtonStep = -F/DF

        newtonOutOfRange = ((root - xh)*DF - F) * ((root - xl)*DF - F) >= 0
        newtonDecreasingSlowly = np.abs(2.*F) > np.abs(dxOld*DF)
        dxOld = dx
        xOld = root
        root = np.where(newtonOutOfRange | newtonDecreasingSlowly,
                        xl + bisectStep,
                        root + newtonStep)
        dx = root - xOld

        F, DF = f_and_fprime(root)
        
        # MAINTAIN THE BRACKET ON THE ROOT
        xl,xh = lax.cond(F < 0,
                         lambda rt, lo, hi: (rt, hi),
                         lambda rt, lo, hi: (lo, rt),
                         root, xl, xh)

        i += 1
        #converged = (np.abs(dx) < x_tol*root) # relative tolerance
        converged = (np.abs(dx) < x_tol) # absolute tolerance
        return root, dx, dxOld, F, DF, xl, xh, converged, i

    x, _, _, _, _, _, _, converged, iters = while_loop(cond,
                                                       loop_body,
                                                       (x0, dx, dxOld, F, DF, xl, xh, False, 0))
    
    return x#, SolutionInfo(converged=converged, function_calls=functionCalls,
             #              iterations=iters)

