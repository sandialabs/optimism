from optimism.JaxConfig import *
from optimism import Surface
from optimism.contact import EdgeIntersection
from optimism import QuadratureRule


def get_coords_and_distances(mesh, ray, edge, neighbor):
    neighborCoords = Surface.get_coords(mesh, neighbor)
    distance, parCoord = EdgeIntersection.compute_valid_ray_trace_distance(neighborCoords, ray)
    distance = if_then_else(np.all(neighbor==edge), np.inf, distance)
    return np.array([distance, parCoord])


def get_best_neighbor(mesh, edge, listOfEdges, normal, x):
    ray = np.array([x, normal])
    coordsAndDists = vmap(get_coords_and_distances, (None,None,None,0))(mesh, ray, edge, listOfEdges)
    bestNeighbor = np.argmin(coordsAndDists[:,0])
    return np.array([bestNeighbor, coordsAndDists[bestNeighbor,1]])
    

def get_best_neighbors(mesh, quadRule, listOfEdges, edge):
    edgeCoords = Surface.get_coords(mesh, edge)
    normal = Surface.compute_normal(edgeCoords)
    edgeLocations = QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords)    
    return vmap(get_best_neighbor, (None, None, None, None, 0))(mesh, edge, listOfEdges, normal, edgeLocations)


#@jit
def construct_edge_neighbor_map(mesh, quadRule, edges):
    return vmap(get_best_neighbors, (None, None, None, 0))(mesh, quadRule, edges, edges)

