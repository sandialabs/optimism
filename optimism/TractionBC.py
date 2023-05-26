import jax
import jax.numpy as np

from optimism import FunctionSpace
from optimism import Mesh


def compute_traction_potential_energy_on_edge(fs, U, quadRule, edge, load, time):
    uq = FunctionSpace.interpolate_nodal_field_on_edge(fs, U, quadRule.xigauss, edge)
    Xq = FunctionSpace.interpolate_nodal_field_on_edge(fs, fs.mesh.coords, quadRule.xigauss, edge)
    edgeCoords = Mesh.get_edge_coords(fs.mesh, edge)
    _, normal, jac = Mesh.compute_edge_vectors(fs.mesh, edgeCoords)
    tq = jax.vmap(load, (0, None, None))(Xq, normal, time)
    integrand = jax.vmap(lambda u,t: u@t)(uq, tq)
    return -np.dot(integrand, jac*quadRule.wgauss)


def compute_traction_potential_energy(fs, U, quadRule, edges, load, time=0.0):
    return np.sum( jax.vmap(compute_traction_potential_energy_on_edge, (None,None,None,0,None,None))(fs, U, quadRule, edges, load, time) )
