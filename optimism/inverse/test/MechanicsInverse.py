from collections import namedtuple

from optimism.JaxConfig import *
from optimism import Mechanics
from optimism import FunctionSpace
from optimism import Interpolants
from optimism.TensorMath import tensor_2D_to_3D

MechanicsInverseFunctions = namedtuple('MechanicsInverseFunctions',
                                      ['ivs_update_jac_ivs_prev',
                                       'ivs_update_jac_disp_vjp',
                                       'ivs_update_jac_coords_vjp',
                                       'nodal_forces_parameterized',
                                       'residual_jac_ivs_prev_vjp',
                                       'residual_jac_coords_vjp'])

def _compute_quadrature_point_field_gradient(u, shapeGrad):
    dg = np.tensordot(u, shapeGrad, axes=[0,0])
    return dg

def _compute_element_field_gradient(U, elemShapeGrads, elemConnectivity, modify_element_gradient):
    elemNodalDisps = U[elemConnectivity]
    elemGrads = vmap(_compute_quadrature_point_field_gradient, (None, 0))(elemNodalDisps, elemShapeGrads)
    elemGrads = modify_element_gradient(elemGrads)
    return elemGrads

def _compute_field_gradient(shapeGrads, conns, nodalField, modify_element_gradient):
    return vmap(_compute_element_field_gradient, (None,0,0,None))(nodalField, shapeGrads, conns, modify_element_gradient)

def _compute_updated_internal_variables_gradient(dispGrads, states, dt, compute_state_new, output_shape):
    dgQuadPointRavel = dispGrads.reshape(dispGrads.shape[0]*dispGrads.shape[1],*dispGrads.shape[2:])
    stQuadPointRavel = states.reshape(states.shape[0]*states.shape[1],*states.shape[2:])
    statesNew = vmap(compute_state_new, (0, 0, None))(dgQuadPointRavel, stQuadPointRavel, dt)
    return statesNew.reshape(output_shape)

def _compute_strain_energy(functionSpace, coords, shapeGrads, vols, 
                           UField, stateField, dt, 
                           compute_energy_density, modify_element_gradient):
    L = Mechanics.strain_energy_density_to_lagrangian_density(compute_energy_density)
    return _integrate_over_block(functionSpace, coords, shapeGrads, vols, 
                                 UField, stateField, dt, 
                                 L, slice(None), modify_element_gradient)

def _integrate_over_block(functionSpace, coords, shapeGrads, vols, 
                          U, stateVars, dt, 
                          func, block, modify_element_gradient):
    vals = _evaluate_on_block(functionSpace, coords, shapeGrads, 
                              U, stateVars, dt, 
                              func, block, modify_element_gradient)
    return np.dot(vals.ravel(), vols[block].ravel())

def _evaluate_on_block(functionSpace, coords, shapeGrads, 
                       U, stateVars, dt, 
                       func, block, modify_element_gradient):
    compute_elem_values = vmap(_evaluate_on_element, (None, None, 0, None, 0, 0, 0, None, None))
    
    blockValues = compute_elem_values(U, coords, stateVars[block], dt, 
                                      functionSpace.shapes[block], shapeGrads[block], functionSpace.mesh.conns[block], 
                                      func, modify_element_gradient)
    return blockValues

def _evaluate_on_element(U, coords, elemStates, dt, 
                         elemShapes, elemShapeGrads, elemConn, 
                         kernelFunc, modify_element_gradient):
    elemVals = FunctionSpace.interpolate_to_element_points(U, elemShapes, elemConn)
    elemGrads = _compute_element_field_gradient(U, elemShapeGrads, elemConn, modify_element_gradient)
    elemXs = FunctionSpace.interpolate_to_element_points(coords, elemShapes, elemConn)
    vmapArgs = 0, 0, 0, 0, None
    fVals = vmap(kernelFunc, vmapArgs)(elemVals, elemGrads, elemStates, elemXs, dt)
    return fVals


def create_mechanics_inverse_functions(functionSpace, createField, mode2D, materialModel, pressureProjectionDegree=None, dt=0.0):
    fs = functionSpace
    shapeOnRef = Interpolants.compute_shapes(fs.mesh.parentElement, fs.quadratureRule.xigauss)

    if mode2D == 'plane strain':
         grad_2D_to_3D = vmap(tensor_2D_to_3D)
    elif mode2D == 'axisymmetric':
        raise NotImplementedError

    modify_element_gradient = grad_2D_to_3D
    if pressureProjectionDegree is not None:
        raise NotImplementedError

    def compute_partial_ivs_update_partial_ivs_prev(U, stateVariables, dt=dt):
        dispGrads = _compute_field_gradient(fs.shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        update_gradient = jacfwd(materialModel.compute_state_new, argnums=1)
        grad_shape = stateVariables.shape + (stateVariables.shape[2],)
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, dt,\
                                                            update_gradient, grad_shape)

    def compute_ivs_update_parameterized(U, stateVariables, coordinates, dt=dt):
        coords = coordinates.reshape(fs.mesh.coords.shape)
        shapeGrads = vmap(FunctionSpace.map_element_shape_grads, (None, 0, None, None))(coords, fs.mesh.conns, fs.mesh.parentElement, shapeOnRef.gradients)
        dispGrads = _compute_field_gradient(shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        update_func = materialModel.compute_state_new
        output_shape = stateVariables.shape
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, dt,\
                                                            update_func, output_shape)

    compute_partial_ivs_update_partial_coords = jit(lambda u, ivs, x, av: 
                                                    vjp(lambda z: compute_ivs_update_parameterized(u, ivs, z), x)[1](av)[0])

    def compute_ivs_update(U, stateVariables, dt=dt):
        dispGrads = _compute_field_gradient(fs.shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        update_func = materialModel.compute_state_new
        output_shape = stateVariables.shape
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, dt,\
                                                            update_func, output_shape)

    compute_partial_ivs_update_partial_disp = jit(lambda x, ivs, av: 
                                                  vjp(lambda z: compute_ivs_update(z, ivs), x)[1](av)[0])

    def compute_strain_energy_parameterized(U, stateVariables, coordinates, dt=dt):
        coords = coordinates.reshape(fs.mesh.coords.shape)
        shapes = vmap(lambda elConns, elShape: elShape, (0, None))(fs.mesh.conns, shapeOnRef.values)
        vols = vmap(FunctionSpace.compute_element_volumes, (None, 0, None, 0, None))(coords, fs.mesh.conns, fs.mesh.parentElement, shapes, fs.quadratureRule.wgauss)
        shapeGrads = vmap(FunctionSpace.map_element_shape_grads, (None, 0, None, None))(coords, fs.mesh.conns, fs.mesh.parentElement, shapeOnRef.gradients)
        return _compute_strain_energy(fs, coords, shapeGrads, vols, U, stateVariables, dt, materialModel.compute_energy_density, modify_element_gradient)

    def compute_strain_energy_for_residual(Uu, p, stateVariables, coordinates, dt=dt):
        U = createField(Uu, p)
        return compute_strain_energy_parameterized(U, stateVariables, coordinates, dt)

    compute_partial_residual_partial_ivs_prev = jit(lambda u, q, iv, x, vx: 
                                                    vjp(lambda z: grad(compute_strain_energy_for_residual, 0)(u, q, z, x), iv)[1](vx)[0])

    compute_partial_residual_partial_coords = jit(lambda u, q, iv, x, vx: 
                                                  vjp(lambda z: grad(compute_strain_energy_for_residual, 0)(u, q, iv, z), x)[1](vx)[0])

    return MechanicsInverseFunctions(jit(compute_partial_ivs_update_partial_ivs_prev), 
                                     compute_partial_ivs_update_partial_disp,
                                     compute_partial_ivs_update_partial_coords,
                                     jit(grad(compute_strain_energy_parameterized)),
                                     compute_partial_residual_partial_ivs_prev,
                                     compute_partial_residual_partial_coords
                                     )
