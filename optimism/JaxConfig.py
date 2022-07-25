from functools import partial
from collections import namedtuple
import jax.numpy as np
from jax import grad, jacrev, jacfwd, jvp, vjp, vmap, value_and_grad, hessian, ops, lax, make_jaxpr, linearize, jit
from jax import custom_jvp, custom_vjp
from jax.lax import while_loop


def if_then_else(cond, val1, val2):
    return lax.cond(cond,
                    lambda x: val1,
                    lambda x: val2,
                    None)


def hessvec(f):
    return lambda x,v: grad(lambda z: np.vdot(grad(f)(z), v))(x)


def hessrayleigh(f):
    return lambda x,v: np.vdot(v, grad(lambda z: np.vdot(grad(f)(z), v))(x))
