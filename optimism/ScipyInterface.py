import jax.numpy as np
import numpy as onp

def make_scipy_linear_function(linear_function):
    """Transform a linear function of a jax array to one that can be used with scipy.linalg.LinearOperator."""
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