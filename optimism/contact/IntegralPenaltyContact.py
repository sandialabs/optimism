import jax
from jax import numpy as jnp
from typing import Callable

def compute_normal(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = jnp.array([tangent[1], -tangent[0]])
    #normalize = normal @ normal
    normalize = jnp.linalg.norm(normal)
    normalize = jnp.where(normalize < 1e-12, 1.0, normalize)
    return normal / normalize


def compute_normal_from_b(edgeA : jnp.array, edgeB : jnp.array) -> jnp.array:
    return -compute_normal(edgeB)


def smooth_linear(xi, l):                
    return jnp.where( xi < l, 0.5*xi*xi/l, jnp.where(xi > 1.0-l, 1.0-l-0.5*(1.0-xi)*(1.0-xi)/l, xi-0.5*l) ) / (1.0 - l)


def compute_intersection(edgeA, edgeB, normal, smoothing):

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = jnp.array([edgeB[0]-edgeB[1], normal]).T
        detM = jnp.linalg.det(M)
        M = jnp.where((jnp.abs(detM) < 1e-13) | (normal@normal < 0.9), jnp.eye(2,2), M)

        r = jnp.array(edgeB[0]-xa)
        xig = jnp.linalg.solve(M,r)
        return xig[0], xig[1]

    def compute_gap(edgeA, xiA, edgeB, normal):
        xA = edgeA[0] * (1.0-xiA) + edgeA[1] * xiA
        _,gb = compute_xi(xA, edgeB, normal)
        return gb

    xi_minmax_in_a = jnp.zeros(2)
    xi_minmax_in_b = jnp.zeros(2)

    xi_g_b_right = compute_xi(edgeA[0], edgeB, normal)
    xi_g_a_right = compute_xi(edgeB[1], edgeA, -normal)

    xi_g_b_right_inside = xi_g_b_right[0] <= 1.0
    xi_minmax_in_a = xi_minmax_in_a.at[0].set(jnp.where(xi_g_b_right_inside, 0.0, xi_g_a_right[0]))
    xi_minmax_in_b = xi_minmax_in_b.at[0].set(jnp.where(xi_g_b_right_inside, xi_g_b_right[0], 1.0))

    xi_g_b_left = compute_xi(edgeA[1], edgeB, normal)
    xi_g_a_left = compute_xi(edgeB[0], edgeA, -normal)

    xi_g_b_left_inside = xi_g_b_left[0] >= 0.0
    xi_minmax_in_a = xi_minmax_in_a.at[1].set(jnp.where(xi_g_b_left_inside, 1.0, xi_g_a_left[0]))
    xi_minmax_in_b = xi_minmax_in_b.at[1].set(jnp.where(xi_g_b_left_inside, xi_g_b_left[0], 0.0))

    xi_minmax_in_a = jnp.maximum(0.0, jnp.minimum(1.0, xi_minmax_in_a) )
    xi_minmax_in_a = smooth_linear(xi_minmax_in_a, smoothing)

    gaps_a = jax.vmap(compute_gap, (None,0,None,None))(edgeA, xi_minmax_in_a, edgeB, normal)

    return xi_minmax_in_a, gaps_a

# integral_x0^x1 delta / g + g / delta - 2 {where g < delta} dxi
def integrate_gap(xi, g, delta):
    dxi = xi[1] - xi[0]
    dxi_invalid = dxi<=1e-14
    a = (g[1]-g[0]) / jnp.where(dxi_invalid, 1.0, dxi)

    g0_large = g[0] > delta
    g1_large = g[1] > delta

    g0_old = g[0]

    g = jnp.where(g0_large, g.at[0].set(delta), g)
    g = jnp.where(g1_large, g.at[1].set(delta), g)

    a_is_zero = jnp.abs(g[0]-g[1]) < 1e-14
    a = jnp.where(a_is_zero | dxi_invalid, 1.0, a)

    xi = jnp.where(g0_large,
                  xi.at[0].set((delta-g0_old) / a + xi[0]),
                  xi)

    xi = jnp.where(g1_large,
                  xi.at[1].set((delta-g0_old) / a + xi[0]),
                  xi)

    dxi = xi[1] - xi[0]

    xi0 = xi[0]
    xi1 = xi[1]

    intgl = (delta / a) * jnp.log(jnp.abs(g[1]/g[0])) + (0.5 * a * (xi1*xi1 - xi0*xi0) + g[0] * xi1 - g[1] * xi0 ) / delta - 2 * dxi
    gbar = 0.5*(g[0]+g[1])
    intgl = jnp.where(a_is_zero, (delta/gbar+gbar/delta-2) * dxi, intgl)

    any_neg = jnp.any(g <= 0)
    intgl = jnp.where(any_neg, jnp.inf, intgl)
    intgl = jnp.where(dxi==0, 0.0 * intgl, intgl)

    return intgl


def integrate_gap_numeric(xi, g, delta):
    N = 10000
    xig = jnp.linspace(0.5/N, 1.0-0.5/N, N)
    dxi = xi[1] - xi[0]
    w = dxi / N

    def gap(x) :
        return g[0] + x * (g[1] - g[0])
    
    def p(x) : 
        v = gap(x)
        return jnp.where(v < delta, v / delta + delta / v - 2, 0.0)

    return jnp.sum( jax.vmap(p)(xig) ) * w


def integrate_mortar_barrier(edgeA : jnp.array,
                              edgeB : jnp.array,
                              f_common_normal : Callable[[jnp.array,jnp.array],jnp.array],
                              allowed_overlap_dist,
                              relativeSmoothingSize = 1e-8):
    
    normal = f_common_normal(edgeA, edgeB)

    nA = compute_normal(edgeA)
    nB = compute_normal(edgeB)
    this_should_be_positive = -(nA @ nB)
    scaling = jnp.where( this_should_be_positive > 0, this_should_be_positive, 0.0)

    xiA,gA = compute_intersection(edgeA, edgeB, normal, relativeSmoothingSize)

    g = scaling * (gA) + allowed_overlap_dist

    lA = jnp.linalg.norm(edgeA[0] - edgeA[1])

    #def penalty(g):
    #    return np.where(g < 0.0, np.inf,
    #                    np.where(g > allowed_overlap_dist, 
    #                             0.0,
    #                             -(g - allowed_overlap_dist)*(g -allowed_overlap_dist) * np.log(g/allowed_overlap_dist)
    #                            )
    #                    )
    #integralA = lA * (xiA[1] - xiA[0]) * 0.5 * (penalty(g[0]) + penalty(g[1]))

    integral = integrate_gap(xiA, g, allowed_overlap_dist)

    return lA * integral