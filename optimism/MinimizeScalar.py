from collections import namedtuple
from optimism.JaxConfig import *
#from jax.lax import while_loop


Settings = namedtuple('Settings',
                      ['tol', 'max_iters'])


def get_settings(tol=1e-8, max_iters=25):
    return Settings(tol, max_iters)


def minimize_scalar(objective, x0, diffArgs, nondiffArgs, settings):

    if not isinstance(diffArgs, tuple):
        msg = "diffArgs argument to optimism.MinimizeScalar.minimize_scalar must be a tuple, got {}"
        raise TypeError(msg.format(diffArgs))

    if not isinstance(diffArgs, tuple):
        msg = "nondiffArgs argument to optimism.MinimizeScalar.minimize_scalar must be a tuple, got {}"
        raise TypeError(msg.format(nondiffArgs))

    # define function with args
    def F(x): return objective(x, *diffArgs, *nondiffArgs)
    
    G = jacfwd(F)
    GH = value_and_grad(G)

    tol = settings.tol
    max_iters = settings.max_iters
    
    def conditional(carry):
        x, f, g, h, alpha, i = carry
        resNorm = np.abs(g)
        return (resNorm > tol) & (i < max_iters)

    def body(carry):
        x, f, g, h, alpha, i = carry
        
        print("h=",h)
        if h > 0:
            # positive curvature, use newton step
            p = -g/h
            alpha0 = 1.0
            alpha = line_search_backtrack(x, f, g, p, alpha0, F)
        else:
            # negative curvature, use gradient descent
            p = -g
            print("alpha initial=", alpha)
            alpha = line_search_bidirectional(x, f, g, p, alpha, F)

        x += alpha*p
        print("i", i, "x", x)
        g, h = GH(x)
        f = F(x)
        return (x, f, g, h, alpha, i + 1)

    f0 = F(x0)
    df0, ddf0 = GH(x0)
    alpha0 = 1.0
    xStar, f, df, _, _, iters = while_loop(conditional, body, (x0, f0, df0, ddf0, alpha0, 0))

    print("\n----\niters taken=", iters)
    print("df=", df)
    print("objective = ", f)
    return xStar


def line_search_bidirectional(x, f, g, p, alpha, F):
    c = 0.01
    cgp = c*g*p
    initialStepLengthSufficient = F(x + alpha*p) < f + cgp*alpha
    print("fwd track = ", initialStepLengthSufficient)
    if initialStepLengthSufficient:
        alpha = line_search_forwardtrack(x, f, g, p, alpha, F)
    else:
        alpha = line_search_backtrack(x, f, g, p, alpha, F)
    return alpha


def line_search_backtrack(x, f, g, p, alpha, F):
    cutback = 0.2
    c = 0.01
    cgp = c*g*p

    def cond_fun(alphaAndIters):
        alpha, i = alphaAndIters
        return (F(x + alpha*p) > f + cgp*alpha) & (i < 20)

    def body_fun(alphaAndIters):
        alpha, i = alphaAndIters
        alpha *= cutback
        return alpha, i + 1

    alpha, lsIters = while_loop(cond_fun, body_fun, (alpha, 0))
    print("alpha=", alpha)
    return alpha


def line_search_forwardtrack(x, f, g, p, alpha, F):
    growth = 1.0/0.2
    c = 0.01
    cgp = c*g*p

    def cond_fun(alphaAndIters):
        alpha, i = alphaAndIters
        return (F(x + growth*alpha*p) < f + cgp*alpha) & (i < 20)

    def body_fun(alphaAndIters):
        alpha, i = alphaAndIters
        alpha *= growth
        return alpha, i + 1

    alpha, lsIters = while_loop(cond_fun, body_fun, (alpha, 0))
    print("alpha=", alpha)
    return alpha


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
