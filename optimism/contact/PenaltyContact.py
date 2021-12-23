from optimism.JaxConfig import *
from optimism import Mesh
from optimism import QuadratureRule
from optimism import Surface

import numpy as onp

def get_current_coordinates_at_quadrature_points(mesh, dispField, quadRule, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    
    edgeCoords = Surface.eval_field(mesh.coords, fieldIndex)
    edgeDisps = Surface.eval_field(dispField, fieldIndex)
    
    return QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords+edgeDisps)


def evaluate_levelset_on_edge(levelset, mesh, dispField, quadRule, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    
    edgeCoords = Surface.eval_field(mesh.coords, fieldIndex)
    edgeDisps = Surface.eval_field(dispField, fieldIndex)
    
    quadratureCurCoords = QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords+edgeDisps)
    return levelset(quadratureCurCoords)


def compute_edge_penalty_contact_energy(levelset, mesh, dispField, quadRule, edge, stiffness):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)
    
    edgeCoords = Surface.eval_field(mesh.coords, fieldIndex)
    edgeDisps = Surface.eval_field(dispField, fieldIndex)
    
    quadratureCurCoords = QuadratureRule.eval_at_iso_points(quadRule.xigauss, edgeCoords+edgeDisps)
    lsetField = levelset(quadratureCurCoords)
    negativeLsetField = np.minimum(0.0, lsetField)
    return stiffness*Surface.integrate_values(quadRule, edgeCoords, np.square(negativeLsetField))


def evaluate_contact_constraints(levelset, dispField, mesh, quadRule, edges):
    return vmap(evaluate_levelset_on_edge, (None,None,None,None,0))(levelset, mesh, dispField, quadRule, edges)


def compute_total_penalty_contact_energy(levelset, dispField, mesh, quadRule, edges, stiffness):
    return np.sum(vmap(compute_edge_penalty_contact_energy, (None,None,None,None,0,None))(levelset, mesh, dispField, quadRule, edges, stiffness))


def compute_fisher_burmeister_linearization(levelset, disp, mesh, quadRule, edges, lmbda):
    for edge in edges:
        quadratureCurCoords = get_current_coordinates_at_quadrature_points(mesh, dispField, quadRule, edge)
        phi = levelset(quadratureCurCoords)

        t = np.sqrt(phi**2 + lmbda**2)

        dlambda = ( t**2 - t*(phi+lmbda) ) / (t-lmbda)

