from functools import partial
from collections import namedtuple
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import jax.numpy as np
    from jax import grad, jacrev, jacfwd, jvp, vjp, value_and_grad, hessian, ops, lax, make_jaxpr, linearize
    from jax import custom_jvp, custom_vjp
    from jax import jit as jaxJit
    from jax import vmap as jaxVmap
    from jax.lax import while_loop
    from jax.config import config

    
config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)


jaxDebug=False

if jaxDebug:
    def jit(f,static_argnums=None):
        return f
    
    vmap = jaxVmap

    def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val
else:
    jit = jaxJit
    vmap = jaxVmap


def if_then_else(cond, val1, val2):
    return lax.cond(cond,
                    lambda x: val1,
                    lambda x: val2,
                    None)


def hessvec(f):
    return lambda x,v: grad(lambda z: np.vdot(grad(f)(z), v))(x)


def hessrayleigh(f):
    return lambda x,v: np.vdot(v, grad(lambda z: np.vdot(grad(f)(z), v))(x))
