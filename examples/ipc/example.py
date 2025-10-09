from jax import custom_jvp
import jax
import jax.numpy as jnp


@custom_jvp
def f(x):
    return jnp.sin(x)


@f.defjvp
def _f_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = f(x)
    tangent_out = jnp.cos(x) * x_dot
    return primal_out[0], tangent_out[0]


def f_jvp(x, v):
    return _f_jvp(x, v)


# forward over reverse
@custom_jvp
def f_grad(x):
    return jax.grad(f)(x)



@f_grad.defjvp
def _f_hvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = f_grad(x)
    tangent_out = -jnp.sin(x) * x_dot
    return primal_out[0], tangent_out[0]

def f_hvp(x, v):
    return jax.jvp(f_grad, x, v)

def f_hess(x):
    return jax.hessian(f)((x,))


x = jnp.array([jnp.sqrt(2.)])
v = jnp.ones(1)
print(f(x))
print(f_grad(x))
# print(jax.grad(f)(x))
# print(hvp(x, v))
print(f_hvp((x,), (v,)))
print(f_hess(x))