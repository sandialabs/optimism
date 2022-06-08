from optimism.JaxConfig import *

from jax.lax import custom_root

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
        automatically clipped to the nearest bound.
    bracket : sequence of 2 reals (list, tuple, numpy array, etc)
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

    bracket = np.array(bracket) # convert to numpy array if not already
    converged = False
    
    # check that root is bracketed
    fl = f(bracket[0])
    fh = f(bracket[1])
    functionCalls = 2

    x0 = np.where(fl*fh < 0,
                  x0,
                  np.nan)

    leftBracketIsSolution = (fl == 0.0)
    x0 = np.where(leftBracketIsSolution, bracket[0], x0)
    converged = np.where(leftBracketIsSolution, True, converged)

    rightBracketIsSolution = (fh == 0.0)
    x0 = np.where(fh == 0.0, bracket[1], x0)
    converged = np.where(rightBracketIsSolution, True, converged)

    # ORIENT THE SEARCH SO THAT F(XL) < 0.
    xl, xh = lax.cond(fl < 0,
                      lambda _: (bracket[0], bracket[1]),
                      lambda _: (bracket[1], bracket[0]),
                      None)

    # INITIALIZE THE GUESS FOR THE ROOT, THE ''STEP SIZE
    # BEFORE LAST'', AND THE LAST STEP
    x0 = np.clip(x0, np.min(bracket), np.max(bracket))
    dxOld = np.abs(bracket[1] - bracket[0])
    dx = dxOld

    F, DF = f_and_fprime(x0)
    functionCalls += 1

    def cond(carry):
        root, dx, dxOld, F, DF, xl, xh, converged, i = carry
        keepLooping = (~converged) & (i < max_iters)
        return keepLooping

    def loop_body(carry):
        root, dx, dxOld, F, DF, xl, xh, converged, i = carry
        
        newtonOutOfRange = ((root - xh)*DF - F) * ((root - xl)*DF - F) > 0
        newtonDecreasingSlowly = np.abs(2.*F) > np.abs(dxOld*DF)
        dxOld = dx
        root, dx, converged = lax.cond(newtonOutOfRange | newtonDecreasingSlowly,
                                       bisection_step,
                                       newton_step,
                                       root, xl, xh, DF, F)

        F, DF = f_and_fprime(root)

        # MAINTAIN THE BRACKET ON THE ROOT
        xl,xh = lax.cond(F < 0,
                         lambda rt, lo, hi: (rt, hi),
                         lambda rt, lo, hi: (lo, rt),
                         root, xl, xh)
        i += 1
        converged = converged | (np.abs(dx) < x_tol) # absolute tolerance
        return root, dx, dxOld, F, DF, xl, xh, converged, i

    x, _, _, _, _, _, _, converged, iters = while_loop(cond,
                                                       loop_body,
                                                       (x0, dx, dxOld, F, DF, xl, xh, converged, 0))

    x = np.where(converged, x, np.nan)
    
    return x#, SolutionInfo(converged=converged, function_calls=functionCalls,
             #              iterations=iters)


def bisection_step(x, xl, xh, df, f):
    dx = 0.5*(xh - xl)
    x = xl + dx
    converged = (x == xl)
    return x, dx, converged


def newton_step(x, xl, xh, df, f):
    dx = -f/df
    temp = x
    x = x + dx
    converged = (x == temp)
    return x, dx, converged
