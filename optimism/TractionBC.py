import jax
import jax.numpy as np

from optimism import Interpolants


def interpolate_nodal_field_on_edge(functionSpace, U, quadRule, edge):
    elemOrdinal = edge[0]
    sideOrdinal = edge[1]
    elemNodeOrdinals = functionSpace.mesh.parentElement.faceNodes[sideOrdinal, :]
    fieldIndex = functionSpace.mesh.conns[elemOrdinal, elemNodeOrdinals]
    edgeShapes = Interpolants.compute_shapes(functionSpace.mesh.parentElement1d, quadRule.xigauss)
    edgeU = U[fieldIndex, :]
    return edgeShapes.values.T@edgeU

def get_edge_coords(mesh, edge):
    edgeNodes = mesh.parentElement.faceNodes[edge[1], :]
    nodes = mesh.conns[edge[0], edgeNodes]
    return mesh.coords[nodes, :]

def compute_edge_vectors(mesh, edgeCoords):
    Xv = edgeCoords[mesh.parentElement1d.vertexNodes, :]
    tangent = Xv[1] - Xv[0]
    normal = np.array([tangent[1], -tangent[0]])
    jac = np.linalg.norm(tangent)
    return tangent/jac, normal/jac, jac

def compute_traction_potential_energy_on_edge(fs, U, quadRule, edge, load, time):
    uq = interpolate_nodal_field_on_edge(fs, U, quadRule, edge)
    Xq = interpolate_nodal_field_on_edge(fs, fs.mesh.coords, quadRule, edge)
    edgeCoords = get_edge_coords(fs.mesh, edge)
    _, normal, jac = compute_edge_vectors(fs.mesh, edgeCoords)
    tq = jax.vmap(load, (0, None, None))(Xq, normal, time)
    integrand = jax.vmap(lambda u,t: u@t)(uq, tq)
    return -np.dot(integrand, jac*quadRule.wgauss)


def compute_traction_potential_energy(fs, U, quadRule, edges, load, time=0.0):
    return np.sum( jax.vmap(compute_traction_potential_energy_on_edge, (None,None,None,0,None,None))(fs, U, quadRule, edges, load, time) )
