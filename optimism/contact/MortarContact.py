import jax
from jax import numpy as jnp
from optimism import QuadratureRule
from optimism import Mesh
from typing import Callable
from functools import partial

# some normal utilities

def compute_normal(edgeCoords):
    tangent = edgeCoords[1]-edgeCoords[0]
    normal = jnp.array([tangent[1], -tangent[0]])
    return normal / jnp.linalg.norm(normal)


def compute_average_normal(edgeA : jnp.array, edgeB : jnp.array) -> jnp.array:
    nA = compute_normal(edgeA)
    nB = compute_normal(edgeB)
    normal = jnp.where(nA != nB, nA - nB, nA)
    return normal / jnp.linalg.norm(normal)


def compute_normal_from_a(edgeA : jnp.array, edgeB : jnp.array) -> jnp.array:
    return compute_normal(edgeA)


def normals_are_facing(edgeA, edgeB):
    return jnp.dot(compute_normal(edgeA), compute_normal(edgeB)) < 0.0

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
                          relativeSmoothingSize : float):
    xiA,xiB,g = compute_intersection(edgeA, edgeB, f_common_normal)
    branches = [lambda : integrate_with_active_mortar(xiA, xiB, g, 
                                                      jnp.linalg.norm(edgeA[0] - edgeA[1]), 
                                                      jnp.linalg.norm(edgeB[0] - edgeB[1]),
                                                      func_of_xiA_xiB_g,
                                                      relativeSmoothingSize),
                lambda : 0.0]

    return jax.lax.switch(1*jnp.logical_not(normals_are_facing(edgeA, edgeB)), branches) 


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

            gapAreaLeft = jax.lax.cond(indexA == -1,
                                       lambda : 0.0,
                                       lambda : integrate_with_mortar(coordsSegB, coordsSegA, f_average_normal, lambda xiA, xiB, gap: f_integrand(gap) * (1.0-xiA), 1e-9))
            gapAreaRight = jax.lax.cond(indexA == -1,
                                        lambda : 0.0,
                                        lambda : integrate_with_mortar(coordsSegB, coordsSegA, f_average_normal, lambda xiA, xiB, gap: f_integrand(gap) * xiA, 1e-9))
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

# utilities for setting up mortar contact data structures

def minimum_squared_distance(xs1, xs2):
    dists = jax.vmap( lambda x: jax.vmap( lambda y: (x-y)@(x-y) )(xs1) ) (xs2)    
    return jnp.min(dists)


@partial(jax.jit, static_argnums=(4,))
def get_closest_neighbors(edgeSetA : jnp.array,
                          edgeSetB : jnp.array,
                          mesh : Mesh.Mesh,
                          disp : jnp.array,
                          maxNeighbors : int):
    def min_dist_squared(edge1, edge2, coords, disp):
        xs1 = coords[edge1] + disp[edge1]
        xs2 = coords[edge2] + disp[edge2]
        return minimum_squared_distance(xs1, xs2)
    
    return _neighbor_search(edgeSetA, edgeSetB, mesh, disp, maxNeighbors, min_dist_squared)


def edges_are_adjacent_non_pacman(edgeA, edgeB, xA, xB):
    case0 = edgeA[1] == edgeB[0]
    case1 = edgeA[0] == edgeB[1]
    x0 = jnp.where(case0, xA, xB)
    x1 = jnp.where(case0, xB, xA)
    edge0_dir = x0[1]-x0[0]
    edge1_norm = compute_normal(x1)
    is_pacman = edge0_dir @ edge1_norm < 0.0
    return ~is_pacman & (case0 | case1)


@partial(jax.jit, static_argnums=(4,5))
def get_closest_neighbors_for_self_contact(edgeSetA : jnp.array,
                                           edgeSetB : jnp.array,
                                           mesh : Mesh.Mesh,
                                           disp : jnp.array,
                                           maxNeighbors : int,
                                           maxPenetrationDistance : float):
    def edges_are_adjacent(edge1, edge2):
        return (edge1[0] == edge2[0]) | \
               (edge1[0] == edge2[1]) | \
               (edge1[1] == edge2[0]) | \
               (edge1[1] == edge2[1])
    
    def edge_A_is_penetrating_beyond_max_distance(xsA, xsB, minSquaredDistance):
        def edge_A_is_penetrating(xsA, xsB):
            normalB = compute_normal(xsB)
            dists = xsA - xsB
            return jnp.any(jnp.dot(dists, normalB) < 0.0)
        return jnp.logical_and(edge_A_is_penetrating(xsA, xsB),
                               minSquaredDistance > maxPenetrationDistance**2)

    def min_dist_conditional(edgeA, edgeB, coords, disp):
        xsA = coords[edgeA] + disp[edgeA]
        xsB = coords[edgeB] + disp[edgeB]
        minSquaredDistance = minimum_squared_distance(xsA, xsB)
        excludeEdges = jnp.logical_or(edges_are_adjacent_non_pacman(edgeA, edgeB, xsA, xsB),
                                      edge_A_is_penetrating_beyond_max_distance(xsA, xsB, minSquaredDistance))
        return jax.lax.cond(excludeEdges, 
                            lambda e1, e2: jnp.inf, 
                            lambda e1, e2: minSquaredDistance, 
                            edgeA, edgeB)
    
    return _neighbor_search(edgeSetA, edgeSetB, mesh, disp, maxNeighbors, min_dist_conditional)


def _neighbor_search(edgeSetA : jnp.array,
                     edgeSetB : jnp.array,
                     mesh : Mesh.Mesh,
                     disp : jnp.array,
                     maxNeighbors : int,
                     f_min_distance : Callable):
    
    def get_close_edge_indices(surfaceA, edgeB):
        minDistsToA = jax.vmap(f_min_distance, (0,None,None,None))(surfaceA, edgeB, mesh.coords, disp)
        sortedEntries = jnp.argsort(minDistsToA)[:maxNeighbors]
        validEntries = jnp.isfinite(minDistsToA)[sortedEntries]
        return jnp.where(validEntries == True, sortedEntries, -1)
    
    return jax.vmap(get_close_edge_indices, (None,0))(edgeSetA, edgeSetB) # loop over surface B, get neighbor index in A


@jax.jit
def get_facet_connectivities(mesh : Mesh.Mesh, sideset):
    def get_sub_segments(side):
        indices = Mesh.get_edge_node_indices(mesh, side)
        return jax.vmap(lambda x,y: jnp.array([x,y]))(indices[:-1], indices[1:])

    segmentConns = jax.vmap(get_sub_segments)(sideset)
    return segmentConns.reshape((segmentConns.shape[0]*segmentConns.shape[1], segmentConns.shape[2]))