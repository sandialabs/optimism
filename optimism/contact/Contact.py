from optimism.JaxConfig import *

from optimism import Surface
from optimism.contact import EdgeIntersection
from optimism.contact import EdgeCpp
from optimism import QuadratureRule
from optimism.contact import Friction
from optimism.contact.SmoothMinMax import min as smooth_min


# surfaceI means integration side/surface/face (subordinate)
# surfaceM means main side/surface/face (geometrically controlling)

def compute_closest_distance_to_each_side(mesh, disp, quadRule, interactionList, surfaceI, smoothingTol=None):
    return vmap(compute_projection_dists, (None,None,None,0,0))(mesh, disp, quadRule, interactionList, surfaceI)


def compute_closest_distance_to_each_side_smooth(mesh, disp, quadRule, interactionList, surfaceI, smoothingTol=1e-5):
    return vmap(compute_projection_dists_smooth, (None,None,None,0,0,None))(mesh, disp, quadRule, interactionList, surfaceI, smoothingTol)


def get_potential_interaction_list(surfaceM, surfaceI, mesh, disp, maxNeighbors):

    def get_close_edge_indices(surfaceM, edgeI):
        minDistsToA = vmap(min_dist_squared, (0,None,None,None,None))(surfaceM, edgeI, mesh, mesh.coords, disp)
        return surfaceM[np.argsort(minDistsToA)[:maxNeighbors]]
    
    return vmap(get_close_edge_indices, (None,0))(surfaceM, surfaceI)


def compute_closest_edges_and_field_weights(mesh, disp, quadRule, interactionList, surfaceI):

    def get_closest_edge(coordsM, edgesM, point):
        cppDists = vmap(EdgeCpp.cpp_distance, (0,None))(coordsM, point)
        i = np.argmin( np.abs(cppDists) )
        return edgesM[i]


    # edgesM here is maxNeighbors per integration edge
    def get_closest_edges(mesh, dispField, quadRule, edgesM, edgeI):
        coordsI = get_side_coordinates(mesh, dispField, edgeI)
        coordsQ = QuadratureRule.eval_at_iso_points(quadRule.xigauss, coordsI)
        coordsM = vmap(get_side_coordinates, (None,None,0))(mesh, dispField, edgesM)
        return vmap(get_closest_edge, (None,None,0))(coordsM, edgesM, coordsQ)


    # edgesM here is 1 per quadrature point
    def get_edge_weights(mesh, dispField, quadRule, edgesM, edgeI):
        coordsI = get_side_coordinates(mesh, dispField, edgeI)
        coordsM = vmap(get_side_coordinates, (None,None,0))(mesh, dispField, edgesM)
        coordsQ = QuadratureRule.eval_at_iso_points(quadRule.xigauss, coordsI)
        return vmap(lambda edge, p: EdgeCpp.cpp_line(edge,p)[1])(coordsM, coordsQ)


    closestEdges = vmap(get_closest_edges, (None,None,None,0,0))(mesh, disp, quadRule, interactionList, surfaceI)
    edgeWeights = vmap(get_edge_weights, (None,None,None,0,0))(mesh, disp, quadRule, closestEdges, surfaceI)
    return closestEdges, edgeWeights


def compute_q_coordinates_from_field_weights(mesh, disp, closestEdges, edgeWeights):

    def compute_coordinate(edgeM, weight):
        coordM = get_side_coordinates(mesh, disp, edgeM)
        return coordM[0] * (1.0-weight) + coordM[1] * weight
    
    def compute_coordinates(closestEdgesQ, edgeWeightsQ):
        return vmap(compute_coordinate)(closestEdgesQ, edgeWeightsQ)

    return vmap(compute_coordinates)(closestEdges, edgeWeights)


def compute_q_coordinates(mesh, disp, quadRule, surfaceI):
    
    def compute_coordinates(edge): 
        edgeCoords = get_side_coordinates(mesh, disp, edge)
        return QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords)

    return vmap(compute_coordinates)(surfaceI)


def compute_friction_potential(mesh, disp, lam, frictionParams,
                               quadRule, surfI, closestSides, sideWeights):

    # the relative motion is assuming that the weights applied to the closest sides at the
    # previous timestep were exactly co-located with the location of the quadrature points
    # at the previous timestep
    
    def compute_friction_potential_on_edge(relativeMotionOnEdge, lamI, edgeI):
        coordI = get_side_coordinates(mesh, disp, edgeI)
        normal = Surface.compute_normal(coordI)
        projection = np.identity(2) - np.outer(normal, normal)    
        sPerps = vmap(lambda x : projection@x)(relativeMotionOnEdge)
        energies = vmap(lambda sPerp,lam: Friction.compute_friction_energy_from_perp_slip(sPerp, frictionParams)*lam)(sPerps,lamI)
        return np.sum(energies)


    relativeMotion = compute_q_coordinates_from_field_weights(mesh, disp, closestSides, sideWeights) - \
        compute_q_coordinates(mesh, disp, quadRule, surfI)
    
    energies = vmap(compute_friction_potential_on_edge,(0,0,0))(relativeMotion, lam, surfI)
    
    return np.sum(energies)


#### Internal functions


def min_dist_squared(edge1, edge2, mesh, coords, disp):
    index1 = Surface.get_field_index(edge1, mesh.conns)
    index2 = Surface.get_field_index(edge2, mesh.conns)

    xs1 = Surface.eval_field(coords, index1) + Surface.eval_field(disp, index1)
    xs2 = Surface.eval_field(coords, index2) + Surface.eval_field(disp, index2)
    
    dists = vmap( lambda x: vmap( lambda y: (x-y)@(x-y) )(xs1) ) (xs2)    
    return np.amin(dists)


def get_side_coordinates(mesh, dispField, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    return Surface.eval_field(mesh.coords, fieldIndex) + Surface.eval_field(dispField, fieldIndex)


def get_closest_distance(coordsM, point):
    cppDists = vmap(EdgeCpp.cpp_distance, (0,None))(coordsM, point)
    i = np.argmin( np.abs(cppDists) )
    return cppDists[i]


def compute_projection_dists(mesh, dispField, quadRule, edgesM, edgeI):
    coordsI = get_side_coordinates(mesh, dispField, edgeI)
    coordsQ = QuadratureRule.eval_at_iso_points(quadRule.xigauss, coordsI)
    coordsM = vmap(get_side_coordinates, (None,None,0))(mesh, dispField, edgesM)
    return vmap(get_closest_distance, (None,0))(coordsM, coordsQ)


def get_closest_two_edges(coordsM, point):        
    cppDists = vmap(EdgeCpp.cpp_distance, (0,None))(coordsM, point)
    sortedIndices = np.argsort( np.abs(cppDists) )
    return coordsM[sortedIndices[:2]]


def get_closest_distance_smooth(coordsM, point, smoothingTol):
    twoClosestEdges = get_closest_two_edges(coordsM, point)
    return EdgeCpp.smooth_distance(twoClosestEdges, point, smoothingTol)


def compute_projection_dists_smooth(mesh, dispField, quadRule, edgesM, edgeI, smoothingTol):
    coordsI = get_side_coordinates(mesh, dispField, edgeI)
    coordsQ = QuadratureRule.eval_at_iso_points(quadRule.xigauss, coordsI)    
    coordsM = vmap(get_side_coordinates, (None,None,0))(mesh, dispField, edgesM)
    return vmap(get_closest_distance_smooth, (None,0,None))(coordsM, coordsQ, smoothingTol)
