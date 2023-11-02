from collections import namedtuple

from optimism.JaxConfig import *
from optimism import Mechanics
from optimism import FunctionSpace

MechanicsInverseFunctions = namedtuple('MechanicsInverseFunctions',
                                      ['partial_ivs_update_partial_ivs_prev'])

def _compute_updated_internal_variables_gradient(dispGrads, states, dt, compute_state_new, gradient_shape):
    dgQuadPointRavel = dispGrads.reshape(dispGrads.shape[0]*dispGrads.shape[1],*dispGrads.shape[2:])
    stQuadPointRavel = states.reshape(states.shape[0]*states.shape[1],*states.shape[2:])
    statesNew = vmap(compute_state_new, (0, 0, None))(dgQuadPointRavel, stQuadPointRavel, dt)
    return statesNew.reshape(gradient_shape)

def create_mechanics_inverse_functions(functionSpace, mode2D, materialModel, pressureProjectionDegree=None):
    fs = functionSpace

    if mode2D == 'plane strain':
         grad_2D_to_3D = Mechanics.plane_strain_gradient_transformation
    elif mode2D == 'axisymmetric':
        raise NotImplementedError

    modify_element_gradient = grad_2D_to_3D
    if pressureProjectionDegree is not None:
        raise NotImplementedError

    def compute_partial_ivs_update_partial_ivs_prev(U, stateVariables, dt=0.0):
        dispGrads = FunctionSpace.compute_field_gradient(fs, U, modify_element_gradient)
        update_gradient = jacfwd(materialModel.compute_state_new, argnums=1)
        grad_shape = stateVariables.shape + (stateVariables.shape[2],)
        return _compute_updated_internal_variables_gradient(dispGrads, stateVariables, dt,\
                                                            update_gradient, grad_shape)

    return MechanicsInverseFunctions(jit(compute_partial_ivs_update_partial_ivs_prev))
