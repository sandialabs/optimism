import jax
import jax.numpy as np

from optimism import QuadratureRule
from optimism import Surface


def interpolate_nodal_field_on_edge(mesh, U, quadRule, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)  
    nodalValues = Surface.eval_field(U, fieldIndex)    
    return QuadratureRule.eval_at_iso_points(quadRule.xigauss, nodalValues)


def compute_traction_potential_energy_on_edge(mesh, U, quadRule, edge, load, time):
    uq = interpolate_nodal_field_on_edge(mesh, U, quadRule, edge)
    Xq = interpolate_nodal_field_on_edge(mesh, mesh.coords, quadRule, edge)
    edgeCoords = Surface.get_coords(mesh, edge)
    edgeNormal = Surface.compute_normal(edgeCoords)
    tq = jax.vmap(load, (0, None, None))(Xq, edgeNormal, time)
    integrand = jax.vmap(lambda u,t: u@t)(uq, tq)
    return -Surface.integrate_values(quadRule, edgeCoords, integrand)


def compute_traction_potential_energy(mesh, U, quadRule, edges, load, time=0.0):
    return np.sum( jax.vmap(compute_traction_potential_energy_on_edge, (None,None,None,0,None,None))(mesh, U, quadRule, edges, load, time) )
