from collections import namedtuple
import math
import numpy as onp
import scipy.special

import jax.numpy as np
from jax.lax import switch


QuadratureRule = namedtuple('QuadratureRule', ['xigauss', 'wgauss'])
QuadratureRule.__doc__ = """Quadrature rule points and weights.
    A ``namedtuple`` containing ``xigauss``, a numpy array of the
    coordinates of the sample points in the reference domain, and
    ``wgauss``, a numpy array with the weights.
    """

def len(quadRule):
    """Gets the number of points in a quadrature rule."""
    return quadRule.xigauss.shape[0]


def create_quadrature_rule_1D(degree):
    """Creates a Gauss-Legendre quadrature on the unit interval.

    The rule can exactly integrate polynomials of degree up to 
    ``degree``.

    Parameters
    ----------
    degree: Highest degree polynomial to be exactly integrated by the quadrature rule

    Returns
    -------
    A ``QuadratureRule`` named tuple containing the quadrature point coordinates
    and the weights.
    """

    n = math.ceil((degree + 1)/2)
    xi, w = scipy.special.roots_sh_legendre(n)
    return QuadratureRule(xi, w)


def eval_at_iso_points(xigauss, field):
    fields = np.array([field[0,:] + (field[1,:]-field[0,:]) * xi for xi in xigauss])
    return fields


def create_quadrature_rule_on_triangle(degree):
    """Creates a Gauss-Legendre quadrature on the unit triangle.

    The rule can exactly integrate 2D polynomials up to the value of 
    ``degree``. The domain is the triangle between the vertices
    (0, 0)-(1, 0)-(0, 1). The rules here are guaranteed to be 
    cyclically symmetric in triangular coordinates and to have strictly
    positive weights.

    Parameters
    ----------
    degree: Highest degree polynomial to be exactly integrated by the quadrature rule

    Returns
    -------
    A ``QuadratureRule`` named tuple containing the quadrature point coordinates
    and the weights.
    """
    if degree <= 1:
        xi = onp.array([[3.33333333333333333E-01,  3.33333333333333333E-01]])

        w  = onp.array([ 5.00000000000000000E-01 ])
    elif degree == 2:
        xi = onp.array([[6.66666666666666667E-01,  1.66666666666666667E-01],
                        [1.66666666666666667E-01,  6.66666666666666667E-01],
                        [1.66666666666666667E-01,  1.66666666666666667E-01]])

        w  = onp.array([1.66666666666666666E-01,
                        1.66666666666666667E-01,
                        1.66666666666666667E-01])
    elif degree <= 4:
        xi = onp.array([[1.081030181680700E-01,  4.459484909159650E-01],
                        [4.459484909159650E-01,  1.081030181680700E-01],
                        [4.459484909159650E-01,  4.459484909159650E-01],
                        [8.168475729804590E-01,  9.157621350977100E-02],
                        [9.157621350977100E-02,  8.168475729804590E-01],
                        [9.157621350977100E-02,  9.157621350977100E-02]])

        w  = onp.array([1.116907948390055E-01,
                        1.116907948390055E-01,
                        1.116907948390055E-01,
                        5.497587182766100E-02,
                        5.497587182766100E-02,
                        5.497587182766100E-02])
    elif degree <= 5:
        xi = onp.array([[3.33333333333333E-01,  3.33333333333333E-01],
                        [5.97158717897700E-02,  4.70142064105115E-01],
                        [4.70142064105115E-01,  5.97158717897700E-02],
                        [4.70142064105115E-01,  4.70142064105115E-01],
                        [7.97426985353087E-01,  1.01286507323456E-01],
                        [1.01286507323456E-01,  7.97426985353087E-01],
                        [1.01286507323456E-01,  1.01286507323456E-01]])

        w = onp.array([1.12500000000000E-01,
                       6.61970763942530E-02,
                       6.61970763942530E-02,
                       6.61970763942530E-02,
                       6.29695902724135E-02,
                       6.29695902724135E-02,
                       6.29695902724135E-02])
    elif degree <= 6:
        xi = onp.array([[5.01426509658179E-01,  2.49286745170910E-01],
                        [2.49286745170910E-01,  5.01426509658179E-01],
                        [2.49286745170910E-01,  2.49286745170910E-01],
                        [8.73821971016996E-01,  6.30890144915020E-02],
                        [6.30890144915020E-02,  8.73821971016996E-01],
                        [6.30890144915020E-02,  6.30890144915020E-02],
                        [5.31450498448170E-02,  3.10352451033784E-01],
                        [6.36502499121399E-01,  5.31450498448170E-02],
                        [3.10352451033784E-01,  6.36502499121399E-01],
                        [5.31450498448170E-02,  6.36502499121399E-01],
                        [6.36502499121399E-01,  3.10352451033784E-01],
                        [3.10352451033784E-01,  5.31450498448170E-02]])

        w = onp.array([5.83931378631895E-02,
                       5.83931378631895E-02,
                       5.83931378631895E-02,
                       2.54224531851035E-02,
                       2.54224531851035E-02,
                       2.54224531851035E-02,
                       4.14255378091870E-02,
                       4.14255378091870E-02,
                       4.14255378091870E-02,
                       4.14255378091870E-02,
                       4.14255378091870E-02,
                       4.14255378091870E-02])

    return QuadratureRule(xi, w)


def create_padded_quadrature_rule_1D(degree):
    """Creates 1D Gauss quadrature rule data that are padded to maintain a 
    uniform size, which makes this function jit-able. 

    This function is inteded to be used only when jit compilation of calls to the
    quadrature rules are needed. Otherwise, prefer to use the standard quadrature 
    rules. The standard rules do not contain extra 0s for padding, which makes 
    them more efficient when  used repeatedly (such as in the global energy).

    Args:
      degree: degree of highest polynomial to be integrated exactly
    """

    npts = np.ceil((degree + 1)/2).astype(int)
    xi,w = switch(npts,
                  [_gauss_quad_1D_1pt, _gauss_quad_1D_2pt, _gauss_quad_1D_3pt,
                   _gauss_quad_1D_4pt, _gauss_quad_1D_5pt],
                  None)
    return QuadratureRule(0.5*(xi + 1.0), 0.5*w)


def _gauss_quad_1D_1pt(_):
    xi = np.array([0., 0., 0., 0., 0.])
    w  = np.array([2., 0., 0., 0., 0.])
    return xi,w


def _gauss_quad_1D_2pt(_):
    xi = np.array([-0.5773502691896257,  0.5773502691896257,   0.,
                    0.,                  0.])
    w  = np.array([ 1.,                  1.                ,   0.,
                    0.,                 0.])
    return xi,w


def _gauss_quad_1D_3pt(_):
    xi = np.array([-0.7745966692414834,  0.                ,  0.7745966692414834,
                    0.,                  0.])
    w  = np.array([ 0.5555555555555557,  0.8888888888888888,  0.5555555555555557,
                    0.,                  0.])
    return xi,w

def _gauss_quad_1D_4pt(_):
    xi = np.array([-0.8611363115940526 , -0.33998104358485626,  0.33998104358485626,
                    0.8611363115940526 ,  0.])
    w  = np.array([ 0.3478548451374537 ,  0.6521451548625462 ,  0.6521451548625462 ,
                    0.3478548451374537,   0.])
    return xi,w


def _gauss_quad_1D_5pt(_):
    xi = np.array([-0.906179845938664  ,  -0.5384693101056831,  0.                ,  0.5384693101056831,  0.906179845938664  ])
    w  = np.array([ 0.23692688505618942,   0.4786286704993662,  0.568888888888889 ,  0.4786286704993662,  0.23692688505618942])

    return xi,w
