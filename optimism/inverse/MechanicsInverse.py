from collections import namedtuple

from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Interpolants
from optimism.Mechanics import tile_props, vmapPropValue
from optimism.TensorMath import tensor_2D_to_3D

IvsUpdateInverseFunctions = namedtuple('IvsUpdateInverseFunctions',
                                      ['ivs_update_jac_ivs_prev',
                                       'ivs_update_jac_disp_vjp',
                                       'ivs_update_jac_coords_vjp'])

PathDependentResidualInverseFunctions = namedtuple('PathDependentResidualInverseFunctions',
                                                  ['residual_jac_ivs_prev_vjp',
                                                   'residual_jac_coords_vjp'])

ResidualInverseFunctions = namedtuple('ResidualInverseFunctions',
                                     ['residual_jac_coords_vjp',
                                      'residual_jac_disp_jvp'])

def _get_2D_formulation(mode2D):
    if mode2D == 'plane strain':
         return vmap(tensor_2D_to_3D)
    elif mode2D == 'axisymmetric':
        raise NotImplementedError

def _check_pressure_projection(pressureProjectionDegree):
    if pressureProjectionDegree is not None:
        raise NotImplementedError

def _compute_element_field_gradient(U, elemShapeGrads, elemConnectivity, modify_element_gradient):
    elemNodalDisps = U[elemConnectivity]
    elemGrads = vmap(FunctionSpace.compute_quadrature_point_field_gradient, (None, 0))(elemNodalDisps, elemShapeGrads)
    elemGrads = modify_element_gradient(elemGrads)
    return elemGrads

def _compute_field_gradient(shapeGrads, conns, nodalField, modify_element_gradient):
    return vmap(_compute_element_field_gradient, (None,0,0,None))(nodalField, shapeGrads, conns, modify_element_gradient)

def _compute_updated_internal_variables_gradient(dispGrads, states, props, dt, compute_state_new, output_shape):
    dgQuadPointRavel = dispGrads.reshape(dispGrads.shape[0]*dispGrads.shape[1],*dispGrads.shape[2:])
    stQuadPointRavel = states.reshape(states.shape[0]*states.shape[1],*states.shape[2:])
    prop_vmap_axes = vmapPropValue(props) # -> 0 - vmap over all quadrature points for properties or None - don't vmap over quadrature points for properties
    new_props = tile_props(props, dispGrads.shape[0], dispGrads.shape[1]) # -> (n_els * n_quadrature_pts, n_props) or (n_props,)
    statesNew = vmap(compute_state_new, (0, 0, prop_vmap_axes, None))(dgQuadPointRavel, stQuadPointRavel, new_props, dt)
    return statesNew.reshape(output_shape)

def _compute_multi_block_updated_internal_variables_gradient(fs, materialModels, dispGrads, states, props, dt):
    statesNew = np.array(states)
    for blockKey in materialModels:
        elemIds = fs.mesh.blocks[blockKey]
        blockDispGrads = dispGrads[elemIds]
        blockStateVariables = states[elemIds]
        blockProps = props[blockKey]
        update_func = materialModels[blockKey].compute_state_new
        output_shape = blockStateVariables.shape
        blockStatesNew = _compute_updated_internal_variables_gradient(blockDispGrads, blockStateVariables, blockProps, dt,\
                                                                      update_func, output_shape)
        statesNew = statesNew.at[elemIds, :, :blockStatesNew.shape[2]].set(blockStatesNew)
    return statesNew


def create_ivs_update_inverse_functions(functionSpace, mode2D, materialModel, pressureProjectionDegree=None):
    fs = functionSpace
    shapeOnRef = Interpolants.compute_shapes(fs.mesh.parentElement, fs.quadratureRule.xigauss)

    modify_element_gradient = _get_2D_formulation(mode2D)
    _check_pressure_projection(pressureProjectionDegree)

    def compute_partial_ivs_update_partial_ivs_prev(U, stateVariables, props, dt=0.0):
        dispGrads = _compute_field_gradient(fs.shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        update_gradient = jacfwd(materialModel.compute_state_new, argnums=1)
        grad_shape = stateVariables.shape + (stateVariables.shape[2],)
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, props, dt,\
                                                            update_gradient, grad_shape)

    def compute_ivs_update_parameterized(U, stateVariables, props, coords, dt=0.0):
        shapeGrads = vmap(FunctionSpace.map_element_shape_grads, (None, 0, None, None))(coords, fs.mesh.conns, fs.mesh.parentElement, shapeOnRef.gradients)
        dispGrads = _compute_field_gradient(shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        update_func = materialModel.compute_state_new
        output_shape = stateVariables.shape
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, props, dt,\
                                                            update_func, output_shape)

    compute_partial_ivs_update_partial_coords = jit(lambda u, ivs, props, x, av, dt=0.0: 
                                                    vjp(lambda z: compute_ivs_update_parameterized(u, ivs, props, z, dt), x)[1](av)[0])

    def compute_ivs_update(U, stateVariables, props, dt=0.0):
        dispGrads = _compute_field_gradient(fs.shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        update_func = materialModel.compute_state_new
        output_shape = stateVariables.shape
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, props, dt,\
                                                            update_func, output_shape)

    compute_partial_ivs_update_partial_disp = jit(lambda x, ivs, props, av, dt=0.0: 
                                                  vjp(lambda z: compute_ivs_update(z, ivs, props, dt), x)[1](av)[0])

    return IvsUpdateInverseFunctions(jit(compute_partial_ivs_update_partial_ivs_prev), 
                                     compute_partial_ivs_update_partial_disp,
                                     compute_partial_ivs_update_partial_coords
                                    )

def create_ivs_update_inverse_functions_multi_block(functionSpace, mode2D, materialModels, pressureProjectionDegree=None):
    fs = functionSpace
    shapeOnRef = Interpolants.compute_shapes(fs.mesh.parentElement, fs.quadratureRule.xigauss)

    modify_element_gradient = _get_2D_formulation(mode2D)
    _check_pressure_projection(pressureProjectionDegree)

    def compute_partial_ivs_update_partial_ivs_prev(U, stateVariables, props, dt=0.0):
        dispGrads = _compute_field_gradient(fs.shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        stateUpdateGrads = np.zeros(stateVariables.shape + (stateVariables.shape[2],))

        for blockKey in materialModels:
            elemIds = functionSpace.mesh.blocks[blockKey]
            blockDispGrads = dispGrads[elemIds]
            blockStateVariables = stateVariables[elemIds]
            blockProps = props[blockKey]
            update_gradient = jacfwd(materialModels[blockKey].compute_state_new, argnums=1)
            grad_shape = blockStateVariables.shape + (blockStateVariables.shape[2],)
            blockStateUpdateGrads = _compute_updated_internal_variables_gradient(blockDispGrads, blockStateVariables, blockProps, dt,\
                                                                                 update_gradient, grad_shape)
            stateUpdateGrads = stateUpdateGrads.at[elemIds, :, :blockStateVariables.shape[2], :blockStateVariables.shape[2]].set(blockStateUpdateGrads)
        return stateUpdateGrads

    def compute_ivs_update_parameterized(U, stateVariables, props, coords, dt=0.0):
        shapeGrads = vmap(FunctionSpace.map_element_shape_grads, (None, 0, None, None))(coords, fs.mesh.conns, fs.mesh.parentElement, shapeOnRef.gradients)
        dispGrads = _compute_field_gradient(shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        return _compute_multi_block_updated_internal_variables_gradient(fs, materialModels, dispGrads, stateVariables, props, dt)

    compute_partial_ivs_update_partial_coords = jit(lambda u, ivs, props, x, av, dt=0.0: 
                                                    vjp(lambda z: compute_ivs_update_parameterized(u, ivs, props, z, dt), x)[1](av)[0])

    def compute_ivs_update(U, stateVariables, props, dt=0.0):
        dispGrads = _compute_field_gradient(fs.shapeGrads, fs.mesh.conns, U, modify_element_gradient)
        return _compute_multi_block_updated_internal_variables_gradient(fs, materialModels, dispGrads, stateVariables, props, dt)

    compute_partial_ivs_update_partial_disp = jit(lambda x, ivs, props, av, dt=0.0: 
                                                  vjp(lambda z: compute_ivs_update(z, ivs, props, dt), x)[1](av)[0])

    return IvsUpdateInverseFunctions(jit(compute_partial_ivs_update_partial_ivs_prev), 
                                     compute_partial_ivs_update_partial_disp,
                                     compute_partial_ivs_update_partial_coords
                                    )

def create_path_dependent_residual_inverse_functions(energyFunction):

    compute_partial_residual_partial_ivs_prev = jit(lambda u, q, iv, x, vx: 
                                                    vjp(lambda z: grad(energyFunction, 0)(u, q, z, x), iv)[1](vx)[0])

    compute_partial_residual_partial_coords = jit(lambda u, q, iv, x, vx: 
                                                  vjp(lambda z: grad(energyFunction, 0)(u, q, iv, z), x)[1](vx)[0])

    return  PathDependentResidualInverseFunctions(compute_partial_residual_partial_ivs_prev,
                                                  compute_partial_residual_partial_coords)

def create_residual_inverse_functions(energyFunction):

    compute_partial_residual_partial_coords = jit(lambda u, q, x, vx: 
                                                  vjp(lambda z: grad(energyFunction, 0)(u, q, z), x)[1](vx)[0])

    compute_partial_parameterized_residual_partial_disp = jit(lambda u, q, x, vx: 
                                                              jvp(lambda z: grad(energyFunction, 0)(z, q, x), (u,), (vx,))[1])

    return  ResidualInverseFunctions(compute_partial_residual_partial_coords, 
                                     compute_partial_parameterized_residual_partial_disp)