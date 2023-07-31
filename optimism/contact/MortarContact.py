import jax
from jax import numpy as jnp
from optimism import QuadratureRule
from collections.abc import Callable

# some normal utilities

def compute_normal(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = jnp.array([tangent[1], -tangent[0]])
    return normal / jnp.linalg.norm(normal)


def compute_average_normal(edgeA : jnp.array, edgeB : jnp.array) -> jnp.array:
    nA = compute_normal(edgeA)
    nB = compute_normal(edgeB)
    normal = nA - nB
    return normal / jnp.linalg.norm(normal)


def compute_normal_from_a(edgeA : jnp.array, edgeB : jnp.array) -> jnp.array:
    return compute_normal(edgeA)

# field utilities

def eval_linear_field_on_edge(field, xi):
    return field[0] * (1.0 - xi) + field[1] * xi


def smooth_linear(xi, l):                
    return jnp.where( xi < l, 0.5*xi*xi/l, jnp.where(xi > 1.0-l, 1.0-l-0.5*(1.0-xi)*(1.0-xi)/l, xi-0.5*l) )


# some actual mortar integrals

def compute_intersection(edgeA, edgeB, f_common_normal):

    def compute_xi(xa, edgeB, normal):
        # solve xa - xb(xi) + g * normal = 0
        # xb = edgeB[0] * (1-xi) + edgeB[1] * xi
        M = jnp.array([edgeB[0]-edgeB[1], normal]).T
        r = jnp.array(edgeB[0]-xa)
        xig = jnp.linalg.solve(M,r)
        return xig[0], xig[1]

    normal = f_common_normal(edgeA, edgeB)
    xiBs1, gs1 = jax.vmap(compute_xi, (0,None,None))(edgeA, edgeB, normal)
    xiAs2, gs2 = jax.vmap(compute_xi, (0,None,None))(edgeB, edgeA,-normal)

    xiAs = jnp.hstack((jnp.arange(2), xiAs2))
    xiBs = jnp.hstack((xiBs1, jnp.arange(2)))
    gs = jnp.hstack((gs1, gs2))

    xiAgood = jax.vmap(lambda xia, xib: jnp.where((xia >= 0.0) & (xia <= 1.0) & (xib >= 0.0) & (xib <= 1.0), xia, jnp.nan))(xiAs, xiBs)
    argsMinMax = jnp.array([jnp.nanargmin(xiAgood), jnp.nanargmax(xiAgood)])

    return xiAs[argsMinMax], xiBs[argsMinMax], gs[argsMinMax]


def integrate_with_active_mortar(xiA, xiB, g, lengthA, lengthB, func_of_xiA_xiB_g, relativeSmoothingSize):
    edgeQuad = QuadratureRule.create_quadrature_rule_1D(degree=2)
    xiGauss = edgeQuad.xigauss

    quadXiA = jax.vmap(eval_linear_field_on_edge, (None,0))(xiA, xiGauss)
    quadXiB = jax.vmap(eval_linear_field_on_edge, (None,0))(xiB, xiGauss)

    xiAsmooth = smooth_linear(xiA, relativeSmoothingSize)
    xiBsmooth = smooth_linear(xiB, relativeSmoothingSize)
    dxiA = xiAsmooth[1] - xiAsmooth[0]
    dxiB = jnp.abs(xiBsmooth[1] - xiBsmooth[0])

    quadWeightA = lengthA * dxiA * edgeQuad.wgauss
    quadWeightB = lengthB * dxiB * edgeQuad.wgauss
    gs = jax.vmap(eval_linear_field_on_edge, (None,0))(g, xiGauss)

    return jnp.dot(0.5*(quadWeightA+quadWeightB), jax.vmap(func_of_xiA_xiB_g)(quadXiA, quadXiB, gs))


def integrate_with_mortar(edgeA : jnp.array,
                          edgeB : jnp.array, 
                          f_common_normal : Callable[[jnp.array,jnp.array],jnp.array],
                          func_of_xiA_xiB_g : Callable[[float,float,float],float],
                          relativeSmoothingSize : float = 1e-7):
    xiA,xiB,g = compute_intersection(edgeA, edgeB, f_common_normal)
    branches = [lambda : integrate_with_active_mortar(xiA, xiB, g, 
                                                      jnp.linalg.norm(edgeA[0] - edgeA[1]), 
                                                      jnp.linalg.norm(edgeB[0] - edgeB[1]),
                                                      func_of_xiA_xiB_g,
                                                      relativeSmoothingSize),
                lambda : 0.0]

    return jax.lax.switch(1*jnp.any(xiA==jnp.nan), branches) #jax.lax.cond(jnp.any(xiA==jnp.nan), branches[1], branches[0]) #


def assembly_mortar_integral(coords, disp, segmentConnsA, segmentConnsB, neighborList, 
                             f_average_normal : Callable,
                             f_integrand : Callable):
    def compute_nodal_gap_area(segB, neighborSegsA):
        nodeLeft = segB[0]
        nodeRight = segB[1]
        def compute_quantities_for_segment_pair(segB, indexA):
            segA = segmentConnsA[indexA]
            coordsSegB = coords[segB] + disp[segB]
            coordsSegA = coords[segA] + disp[segA]

            gapAreaLeft = integrate_with_mortar(coordsSegB, coordsSegA, f_average_normal, lambda xiA, xiB, gap: f_integrand(gap) * (1.0-xiA), 1e-9)
            gapAreaRight = integrate_with_mortar(coordsSegB, coordsSegA, f_average_normal, lambda xiA, xiB, gap: f_integrand(gap) * xiA, 1e-9)
            return gapAreaLeft, gapAreaRight

        gapAreaLeft, gapAreaRight = jax.vmap(compute_quantities_for_segment_pair, (None,0))(segB, neighborSegsA)
        return nodeLeft, jnp.sum(gapAreaLeft), nodeRight, jnp.sum(gapAreaRight)

    nodesLeft, gapsLeft, nodesRight, gapsRight = jax.vmap(compute_nodal_gap_area)(segmentConnsB, neighborList)

    nodalGapField = jnp.zeros(disp.shape[0])
    nodalGapField = nodalGapField.at[nodesLeft].add(gapsLeft)
    nodalGapField = nodalGapField.at[nodesRight].add(gapsRight)

    return nodalGapField


def assemble_area_weighted_gaps(coords, disp, segmentConnsA, segmentConnsB, neighborList, f_average_normal : Callable):
    return assembly_mortar_integral(coords, disp, segmentConnsA, segmentConnsB, neighborList, f_average_normal, lambda gap : gap)


def assemble_nodal_areas(coords, disp, segmentConnsA, segmentConnsB, neighborList, f_average_normal : Callable):
    return assembly_mortar_integral(coords, disp, segmentConnsA, segmentConnsB, neighborList, f_average_normal, lambda gap : 1.0)