import jax
import jax.numpy as np

def solve(f, df, x0, *params):
    def cond(args):
        x, p = args
        r = f(x, *p)
        return np.linalg.norm(r) > 1e-4
    
    def body(args):
        x, p = args
        R = f(x, *p)
        J = df(x, *p)
        x -= R/J
        return x, p
    
    x, _ = jax.lax.while_loop(cond, body, (x0, params))
    return x


if __name__ == "__main__":
    
    f = lambda x, a, b: a*np.dot(x, x) - b
    df = jax.jacfwd(f)
    a = 1.0
    b = 4.0
    x0 = 5.0
    
    x = solve(f, df, x0, a, b)
    
    print(f"x = {x}")