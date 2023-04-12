from functools import partial
import jax
import jax.numpy as np

@partial(jax.custom_jvp, nondiff_argnums=(0,1))
def solve(f, df, x0, *params):
    def cond(args):
        x, p = args
        r = f(x, *p)
        return np.linalg.norm(r) > 1e-6
    
    def body(args):
        x, p = args
        R = f(x, *p)
        J = df(x, *p)
        x -= R/J
        return x, p
    
    x, _ = jax.lax.while_loop(cond, body, (x0, params))
    return x


@solve.defjvp
def solve_jvp(f, df, primals, tangents):
    x0, *params = primals
    _, *dparams = tangents
    x = solve(f, df, x0, *params)
    J = df(x, *params)
    Jinv = 1.0/J
    dfdp = jax.jvp(f, (x, *params), (0.0, *dparams))[1]
    return x, -Jinv*dfdp


if __name__ == "__main__":
    
    f = lambda x, a, b: a*np.dot(x, x) - b
    df = jax.jacfwd(f)
    a = 1.0
    b = 4.0
    x0 = 5.0
    
    x = solve(f, df, x0, a, b)
    print(f"x = {x}")
    
    dxdb = jax.jacfwd(solve, 4)(f, df, x0, a, b)
    print(f"dxdb = {dxdb}")