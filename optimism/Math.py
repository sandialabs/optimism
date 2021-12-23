from optimism.JaxConfig import *
from jax import custom_jvp

# constants
# Assumes IEEE double precision
_DBL_PRECISION_SIGNIFICAND_BITS = 53

# equivalent to ceil(_DBL_PRECISION_SIGNIFICAND_BITS / 2) as an integer
_SPLIT_S = -(-_DBL_PRECISION_SIGNIFICAND_BITS // 2)
_SPLIT_FACTOR = 1<<_SPLIT_S + 1

@custom_jvp
def safe_sqrt(x):
    return np.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(xt, vt):
    x, = xt
    v, = vt
    f = safe_sqrt(x)
    df = v * lax.cond( x <= 0,
                       lambda x: 0.,
                       lambda x: 0.5/f,
                       x )
    return f, df


def sum2(a):
    """
    Sum a vector to much higher accuracy than numpy.sum.

    Parameters
    ----------
    a : ndarray, with only one axis (shape [n,])

    Returns
    -------
    sum : real
        The sum of the numbers in the array


    This special sum method computes the result as accurate as if 
    computed in quadruple precision.

    Reference:
    T. Ogita, S. M. Rump, and S. Oishi. Accurate sum and dot product.
    SIAM J. Sci. Comput., Vol 26, No 6, pp. 1955-1988.
    doi: 10.1137/030601818
    """
    def f(carry, ai):
        p, sigma = carry
        p, q = _two_sum(p, ai)
        sigma += q
        return (p, sigma), p

    total = 0.0
    c = 0.0
    (total, c), partialSums = lax.scan(f, (total, c), a)
    return total + c


def _two_sum(a, b):
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y


def dot2(x, y):
    """
    Compute inner product of 2 vectors to much higher accuracy than numpy.dot.

    Parameters
    ----------
    x : ndarray, with only one axis (shape [n,])
    y : ndarray, with only one axis (shape [n,])

    Returns
    -------
    dotprod : real
        The inner product of the input vectors.


    This special inner product  method computes the result as accurate
    as if computed in quadruple precision. This algorithm is useful to
    computing objective functions from numerical integration. It avoids
    accumulation of floating point cancellation error that can obscure
    whether an objective function has truly decreased.

    The environment variable setting 
        'XLA_FLAGS = "--xla_cpu_enable_fast_math=false"'
    is critical for this function to work on the CPU. Otherwise, xla
    apparently sets a flag for LLVM that allows unsafe floating point
    optimizations that can change associativity.

    Reference:
    T. Ogita, S. M. Rump, and S. Oishi. Accurate sum and dot product.
    SIAM J. Sci. Comput., Vol 26, No 6, pp. 1955-1988.
    doi: 10.1137/030601818
    """
    def f(carry, xy):
        p, s = carry
        xi, yi = xy
        h, r = _two_product(xi, yi)
        p, q = _two_sum(p, h)
        s = s + (q + r)
        return (p, s), p

    rawTotal = 0.0
    compensation = 0.0
    X = np.column_stack((x,y))
    (rawTotal, compensation), partialSums = lax.scan(f,
                                                     (rawTotal, compensation),
                                                     X)
    return rawTotal + compensation


def _two_product(a, b):
    x = a*b
    a1, a2 = _float_split(a)
    b1, b2 = _float_split(b)
    y = a2*b2 - (((x - a1*b1) - a2*b1) - a1*b2)
    return x, y


def _float_split(a):
    c = _SPLIT_FACTOR*a
    x = c - (c - a)
    y = a - x
    return x, y
