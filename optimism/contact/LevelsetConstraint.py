from optimism.JaxConfig import *
from optimism import Mesh
from optimism import QuadratureRule
from optimism import Surface
from optimism import Math
from optimism.contact import Friction

def compute_edge_levelset_constraints(levelset, mesh, dispField, quadRule, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    edgeCoords = Surface.eval_field(mesh.coords, fieldIndex)
    edgeDisps = Surface.eval_field(dispField, fieldIndex)
    
    quadratureCurCoords = QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords+edgeDisps)
    lsetField = levelset(quadratureCurCoords)

    return lsetField


def compute_levelset_constraints(levelset, disp, mesh, quadRule, edges):
    return vmap(compute_edge_levelset_constraints, (None,None,None,None,0))(levelset, mesh, disp, quadRule, edges)


def compute_contact_point_coords_on_edge(mesh, dispField, quadRule, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    edgeCoords = Surface.eval_field(mesh.coords, fieldIndex)
    edgeDisps = Surface.eval_field(dispField, fieldIndex)
    
    quadratureCurCoords = QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords+edgeDisps)

    return quadratureCurCoords


def compute_contact_point_coordinates(disp, mesh, quadRule, edges):
    return vmap(compute_contact_point_coords_on_edge, (None,None,None,0))(mesh, disp, quadRule, edges)



def compute_friction_potential_on_edge(mesh, dispField, relativeMotionOld, lams, quadRule, edge, frictionParams):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    edgeCoords = Surface.eval_field(mesh.coords, fieldIndex)
    edgeDisps = Surface.eval_field(dispField, fieldIndex)
    normal = Surface.compute_normal(edgeCoords + edgeDisps)
    
    quadratureCurCoords = QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords+edgeDisps)
    relativeMotion = quadratureCurCoords - relativeMotionOld

    projection = np.identity(2) - np.outer(normal, normal)

    sPerps = vmap(lambda x : projection@x)(relativeMotion)    
    energies = vmap(lambda sPerp,lam: Friction.compute_friction_energy_from_perp_slip(sPerp, frictionParams)*lam)(sPerps,lams)     

    # no integral because lagrange multipliers are not tractions, but forces
    return np.sum(energies)


def compute_friction_potential(disp, contactCoordsOld, levelsetMotion, lam, mesh, quadRule, edges, frictionParams):
    relativeMotionOld = contactCoordsOld + levelsetMotion[None,None,:]

    nEdges = edges.shape[0]
    nQuadPts = quadRule.xigauss.shape[0]

    lam = np.reshape(lam, (nEdges, nQuadPts))
        
    frictionPots = vmap(compute_friction_potential_on_edge, (None, None, 0, 0, None, 0, None)) \
        (mesh,
         disp,
         relativeMotionOld,
         lam,
         quadRule,
         edges,
         frictionParams)

    return np.sum(frictionPots)

