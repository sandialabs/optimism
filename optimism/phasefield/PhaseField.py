import numpy as onp

from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Mesh
from optimism import QuadratureRule
from optimism.TensorMath import tensor_2D_to_3D


PhaseFieldFunctions = namedtuple('PhaseFieldFunctions',
                                 ['compute_internal_energy',
                                  'compute_output_energy_densities_and_fluxes',
                                  'compute_strain_energy_densities',
                                  'compute_phase_potential_energy',
                                  'compute_element_stiffnesses',
                                  'compute_block_diagonal_element_stiffnesses',
                                  'compute_initial_state',
                                  'compute_updated_internal_variables',
                                  'compute_constraint_hessian'])

element_hess_func = hessian(FunctionSpace.integrate_element_from_local_field)


def compute_element_stiffness_from_global_fields(U, coords, elemInternals, elemConns,
                                                 elemShapes, elemShapeGrads, elemVols,
                                                 lagrangian_density, modify_element_gradient):
    elemU = U[elemConns,:]
    elemCoords = coords[elemConns,:]
    return element_hess_func(elemU, elemCoords, elemInternals, elemShapes, elemShapeGrads,
                             elemVols, lagrangian_density, modify_element_gradient)


def _compute_element_stiffnesses(U, internalVariables, functionSpace, lagrangian_density,
                                 modify_element_gradient):
    f = vmap(compute_element_stiffness_from_global_fields, (None, None, 0, 0, 0, 0, 0, None, None))
    fs = functionSpace
    return f(U, fs.mesh.coords, internalVariables, fs.mesh.conns, fs.shapes, fs.shapeGrads,
             fs.vols, lagrangian_density, modify_element_gradient)


def _compute_block_diagonal_element_stiffnesses(U, internalVariables, functionSpace, lagrangian_density, modify_element_gradient):
    elementKMats = _compute_element_stiffnesses(U, internalVariables, functionSpace, lagrangian_density, modify_element_gradient)

    print('zeroing cross terms')
    
    def zero_out_cross_terms(eStiff):
        eStiff = eStiff.at[:,:2,:,2].set(0.0)
        return eStiff.at[:,2,:,:2].set(0.0)

    return vmap(zero_out_cross_terms)(elementKMats)


def compute_phase_field_constraint_hessian(Lambda, kappa, constraint, dofManager):
    """Augmented Lagrange Hessian terms for lower bound constrained phase field models.

    Because this is a simple bound constraint, the Hessian terms are diagonal.
    This function returns just this diagonal, which can be added onto the 
    unconstrained Hessian matrix.

    Note that return value is a standard numpy array (with one axis). 
    This is so that it can be added to a sparse matrix without causing an error.
    Jax numpy arrays do not work with scipy sparse arrays.

    Args
    ----
    Lambda: array of constraint forces
    kappa: array of penalty parameters
    constraint: values of lower bound constraint functions 
                (just unknown nodal phases)
    dofManager: dofManager object for current mesh

    Returns
    -------
    d: standard numpy array with one axis representing the constraint 
       part of the AL Hessian
    """
    phasePenaltyStiffness = np.where(Lambda >= constraint*kappa, kappa, 0.0)
    phaseDofIds = dofManager.ids[dofManager.isUnknown[:,2],2]
    phaseUnknownIds = dofManager.dofToUnknown[phaseDofIds]
    d = np.zeros(dofManager.get_unknown_size()).at[phaseUnknownIds].set(phasePenaltyStiffness)
    return onp.array(d)


def unpack_fields_2D(U):
    disp = U[:2]
    phase = U[2]
    return disp, phase


def unpack_gradients_2D(gradU):
    dispGrad = gradU[:2]
    phaseGrad = gradU[2]
    return dispGrad, phaseGrad


def energy_density_to_lagrangian_density(energy_density):
    def L(U, gradU, Q, X):
        disp, phase = unpack_fields_2D(U)
        dispGrad = gradU[:3]
        phaseGrad = gradU[3]
        return energy_density(dispGrad, phase, phaseGrad, Q)
    return L


def plane_strain_gradient(gradU):
    dispGrad2D, phaseGrad2D = unpack_gradients_2D(gradU)
    dispGrad = tensor_2D_to_3D(dispGrad2D)
    phaseGrad = np.hstack((phaseGrad2D,0.0))
    return np.row_stack((dispGrad,phaseGrad))


def plane_strain_element_gradient_transformation(elemGrads, elemShapes, elemVols, elemNodalDofs, elemNodalCoords):
    return vmap(plane_strain_gradient)(elemGrads)


def axisymmetric_gradient(gradU, U, coord):
    dispGrad2D, phaseGrad2D = unpack_gradients_2D(gradU)
    disp, _ = unpack_fields_2D(U)
    dispGrad = tensor_2D_to_3D(dispGrad2D)
    dispGrad = dispGrad.at[2,2].set(disp[0]/coord[0])
    phaseGrad = np.hstack((phaseGrad2D,0.0))
    return np.row_stack((dispGrad, phaseGrad))


def axisymmetric_element_gradient_transformation(elemGrads, elemShapes, elemVols, elemNodalDofs, elemNodalCoords):
    elemPointDofs = elemShapes@elemNodalDofs
    elemPointCoords = elemShapes@elemNodalCoords
    return vmap(axisymmetric_gradient)(elemGrads, elemPointDofs, elemPointCoords)


def create_phasefield_functions(functionSpace, mode2D,
                                materialModel, pressureProjectionDegree=None):
    fs = functionSpace

    if mode2D == 'plane strain':
        modify_element_gradient = plane_strain_element_gradient_transformation
    elif mode2D == 'axisymmetric':
        modify_element_gradient = axisymmetric_element_gradient_transformation
    else:
        raise ValueError("mode2D must be set to 'plane strain' or 'axisymmetric'")

    L = energy_density_to_lagrangian_density(materialModel.compute_energy_density)
    
    def compute_internal_energy(U, Q):
        return FunctionSpace.integrate_over_block(fs, U, Q, L, slice(None), modify_element_gradient=modify_element_gradient)

    L_output = energy_density_to_lagrangian_density(materialModel.compute_output_energy_density)
    L_and_fluxes = value_and_grad(L_output, 1)

    def compute_output_energy_densities_and_stresses(U, Q):
        return FunctionSpace.evaluate_on_block(fs, U, Q, L_and_fluxes, slice(None), modify_element_gradient=modify_element_gradient)

    L_strain = energy_density_to_lagrangian_density(materialModel.compute_strain_energy_density)

    def compute_strain_energy_density(U,Q):
        return FunctionSpace.evaluate_on_block(fs, U, Q, L_strain, slice(None), modify_element_gradient=modify_element_gradient)

    def compute_initial_state():
        return materialModel.compute_initial_state((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 1))

    L_compute_state_new = energy_density_to_lagrangian_density(materialModel.compute_state_new)
    
    def compute_updated_internal_variables(U, Q):
        return FunctionSpace.\
            evaluate_on_block(fs, U, Q, L_compute_state_new, slice(None), modify_element_gradient=modify_element_gradient)

    def compute_element_stiffnesses(U, Q):
        return _compute_element_stiffnesses(U, Q, fs, L, modify_element_gradient)

    
    def compute_block_diagonal_element_stiffnesses(U, Q):
        return _compute_block_diagonal_element_stiffnesses(U, Q, fs, L, modify_element_gradient)

    Lphase = energy_density_to_lagrangian_density(materialModel.compute_phase_potential_density)
    def compute_phase_potential_energy(U, Q):
        return FunctionSpace.integrate_over_block(fs, U, Q, Lphase, slice(None), modify_element_gradient=modify_element_gradient)
    
    return PhaseFieldFunctions(compute_internal_energy,
                               jit(compute_output_energy_densities_and_stresses),
                               jit(compute_strain_energy_density),
                               jit(compute_phase_potential_energy),
                               jit(compute_element_stiffnesses),
                               jit(compute_block_diagonal_element_stiffnesses),
                               compute_initial_state,
                               jit(compute_updated_internal_variables),
                               compute_phase_field_constraint_hessian)
                               

