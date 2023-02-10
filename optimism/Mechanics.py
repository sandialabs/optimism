from collections import namedtuple

from optimism.JaxConfig import *
from optimism import SparseMatrixAssembler
from optimism import FunctionSpace
from optimism import Mesh
from optimism.TensorMath import tensor_2D_to_3D
from optimism import QuadratureRule
from optimism import Interpolants

MechanicsFunctions = namedtuple('MechanicsFunctions',
                                ['compute_strain_energy',
                                 'compute_updated_internal_variables',
                                 'compute_element_stiffnesses',
                                 'compute_output_energy_densities_and_stresses',
                                 'compute_initial_state'])


DynamicsFunctions = namedtuple('DynamicsFunctions',
                               ['compute_algorithmic_energy',
                                'compute_updated_internal_variables',
                                'compute_element_hessians',
                                'compute_output_energy_densities_and_stresses',
                                'compute_output_kinetic_energy',
                                'compute_output_strain_energy',
                                'compute_initial_state',
                                'compute_element_masses', # not used for time integration, provided for convenience (spectral analysis, eg)
                                'predict',
                                'correct'])

NewmarkParameters = namedtuple('NewmarkParameters', ['gamma', 'beta'], defaults=[0.5, 0.25])


def plane_strain_gradient_transformation(elemDispGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
    return vmap(tensor_2D_to_3D)(elemDispGrads)


def volume_average_J_gradient_transformation(elemDispGrads, elemVols, pShapes):
    defGrads = elemDispGrads[:,:2,:2] + np.identity(2)
    Js = np.linalg.det(defGrads)
    
    rhs = np.dot(elemVols*Js, pShapes)
    M = pShapes.T@np.diag(elemVols)@pShapes
    nodalJBars = np.linalg.solve(M, rhs)
    
    JBars = pShapes@nodalJBars
    factors = np.sqrt(JBars/Js)
    defGrads = vmap(lambda c, A: c*A)(factors, defGrads)
    return defGrads - np.identity(2)


def axisymmetric_gradient(dispGrad, disp, coord):
    dispGrad = tensor_2D_to_3D(dispGrad)
    dispGrad = dispGrad.at[2,2].set(disp[0]/coord[0])
    return dispGrad


def axisymmetric_element_gradient_transformation(elemDispGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
    elemPointDisps = elemShapes@elemNodalDisps
    elemPointCoords = elemShapes@elemNodalCoords
    return vmap(axisymmetric_gradient)(elemDispGrads, elemPointDisps, elemPointCoords)


element_hess_func = hessian(FunctionSpace.integrate_element_from_local_field)


def compute_element_stiffness_from_global_fields(U, coords, elInternals, elConn, elShapes, elShapeGrads, elVols, lagrangian_density, modify_element_gradient):
    elDisp = U[elConn,:]
    elCoords = coords[elConn,:]
    return element_hess_func(elDisp, elCoords, elInternals, elShapes, elShapeGrads,
                             elVols, lagrangian_density, modify_element_gradient)


def _compute_element_stiffnesses(U, internals, functionSpace, compute_energy_density, modify_element_gradient):
    L = strain_energy_density_to_lagrangian_density(compute_energy_density)
    f =  vmap(compute_element_stiffness_from_global_fields,
              (None, None, 0, 0, 0, 0, 0, None, None))
    fs = functionSpace
    return f(U, fs.mesh.coords, internals, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols,
             L, modify_element_gradient)


def _compute_strain_energy(functionSpace, UField, stateField,
                           compute_energy_density,
                           modify_element_gradient):
    L = strain_energy_density_to_lagrangian_density(compute_energy_density)
    return FunctionSpace.integrate_over_block(functionSpace, UField, stateField, L,
                                              slice(None),
                                              modify_element_gradient=modify_element_gradient)


def _compute_strain_energy_multi_block(functionSpace, UField, stateField, blockModels,
                                       modify_element_gradient):
    energy = 0.0
    for blockKey in blockModels:
        materialModel = blockModels[blockKey]
        elemIds = functionSpace.mesh.blocks[blockKey]
        
        L = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
        
        blockEnergy = FunctionSpace.integrate_over_block(functionSpace, UField, stateField, L,
                                                         elemIds, modify_element_gradient=modify_element_gradient)
        
        energy += blockEnergy
    return energy


def _compute_updated_internal_variables(functionSpace, U, states, compute_state_new, modify_element_gradient):
    dispGrads = FunctionSpace.compute_field_gradient(functionSpace, U, modify_element_gradient)
    dgQuadPointRavel = dispGrads.reshape(dispGrads.shape[0]*dispGrads.shape[1],*dispGrads.shape[2:])
    stQuadPointRavel = states.reshape(states.shape[0]*states.shape[1],*states.shape[2:])
    statesNew = vmap(compute_state_new)(dgQuadPointRavel, stQuadPointRavel)
    return statesNew.reshape(states.shape)


def _compute_updated_internal_variables_multi_block(functionSpace, U, states, blockModels, modify_element_gradient):
    dispGrads = FunctionSpace.compute_field_gradient(functionSpace, U, modify_element_gradient)

    statesNew = np.array(states)

    # # Store the same number of state variables for every material to make
    # vmapping easy. See comment below in _compute_initial_state_multi_block
    numStateVariables = 1
    for blockKey in blockModels:
        numStateVariablesForBlock = blockModels[blockKey].compute_initial_state().shape[0]
        numStateVariables = max(numStateVariables, numStateVariablesForBlock)
    
    for blockKey in blockModels:
        elemIds = functionSpace.mesh.blocks[blockKey]
        blockDispGrads = dispGrads[elemIds]
        blockStates = states[elemIds]
        
        compute_state_new = blockModels[blockKey].compute_state_new
        
        dgQuadPointRavel = blockDispGrads.reshape(blockDispGrads.shape[0]*blockDispGrads.shape[1],*blockDispGrads.shape[2:])
        stQuadPointRavel = blockStates.reshape(blockStates.shape[0]*blockStates.shape[1],-1)
        blockStatesNew = vmap(compute_state_new)(dgQuadPointRavel, stQuadPointRavel).reshape(blockStates.shape)        
        statesNew = statesNew.at[elemIds, :, :blockStatesNew.shape[2]].set(blockStatesNew)
        

    return statesNew


def _compute_initial_state_multi_block(fs, blockModels):

    numQuadPoints = QuadratureRule.len(fs.quadratureRule)
    # Store the same number of state variables for every material to make
    # vmapping easy.
    #
    # TODO(talamini1): Fix this so that every material only stores what it
    # needs and doesn't waste memory.
    #
    # To do this, walk through each material and query number of state
    # variables. Use max to allocate the global state variable array.
    numStateVariables = 1
    for blockKey in blockModels:
        numStateVariablesForBlock = blockModels[blockKey].compute_initial_state().shape[0]
        numStateVariables = max(numStateVariables, numStateVariablesForBlock)

    initialState = np.zeros((Mesh.num_elements(fs.mesh), numQuadPoints, numStateVariables))
    
    for blockKey in blockModels:
        elemIds = fs.mesh.blocks[blockKey]
        state = blockModels[blockKey].compute_initial_state()
        blockInitialState = np.tile(state, (elemIds.size, numQuadPoints, 1))
        initialState = initialState.at[elemIds, :, :blockInitialState.shape[2]].set(blockInitialState)

    return initialState


def _compute_element_stiffnesses_multi_block(U, stateVariables, functionSpace, blockModels, modify_element_gradient):
    fs = functionSpace
    nen = Interpolants.num_nodes(functionSpace.mesh.parentElement)
    elementHessians = np.zeros((Mesh.num_elements(functionSpace.mesh), nen, 2, nen, 2))
    for blockKey in blockModels:
        materialModel = blockModels[blockKey]
        L = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
        elemIds = functionSpace.mesh.blocks[blockKey]
        f =  vmap(compute_element_stiffness_from_global_fields,
                  (None, None, 0, 0, 0, 0, 0, None, None))
        blockHessians = f(U, fs.mesh.coords, stateVariables[elemIds], fs.mesh.conns[elemIds],
                          fs.shapes[elemIds], fs.shapeGrads[elemIds], fs.vols[elemIds],
                          L, modify_element_gradient)
        elementHessians = elementHessians.at[elemIds].set(blockHessians)
    return elementHessians


def strain_energy_density_to_lagrangian_density(strain_energy_density):
    def L(U, gradU, Q, X):
        return strain_energy_density(gradU, Q)
    return L


def create_multi_block_mechanics_functions(functionSpace, mode2D, materialModels, pressureProjectionDegree=None):
    fs = functionSpace

    if mode2D == 'plane strain':
         grad_2D_to_3D = plane_strain_gradient_transformation
    elif mode2D == 'axisymmetric':
        raise NotImplementedError

    modify_element_gradient = grad_2D_to_3D
    if pressureProjectionDegree is not None:
        masterJ = Interpolants.make_master_tri_element(degree=pressureProjectionDegree)
        xigauss = functionSpace.quadratureRule.xigauss
        shapesJ = Interpolants.compute_shapes_on_tri(masterJ, xigauss)

        def modify_element_gradient(elemGrads, elemVols):
            elemGrads = volume_average_J_gradient_transformation(elemGrads, elemVols, shapesJ)
            return grad_2D_to_3D(elemGrads, elemVols)
    
    
    def compute_strain_energy(U, stateVariables):
        return _compute_strain_energy_multi_block(fs, U, stateVariables, materialModels, modify_element_gradient)

        
    def compute_updated_internal_variables(U, stateVariables):
        return _compute_updated_internal_variables_multi_block(fs, U, stateVariables, materialModels, modify_element_gradient)

    
    def compute_element_stiffnesses(U, stateVariables):
        return _compute_element_stiffnesses_multi_block(U, stateVariables, functionSpace, materialModels, modify_element_gradient)


    def compute_output_energy_densities_and_stresses(U, stateVariables): 
        energy_densities = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule)))
        stresses = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 3, 3))
        for blockKey in materialModels:
            compute_output_energy_density = materialModels[blockKey].compute_energy_density
            output_lagrangian = strain_energy_density_to_lagrangian_density(compute_output_energy_density)
            output_constitutive = value_and_grad(output_lagrangian, 1)
            elemIds = fs.mesh.blocks[blockKey]
            blockEnergyDensities, blockStresses = FunctionSpace.evaluate_on_block(fs, U, stateVariables, output_constitutive, elemIds, modify_element_gradient=modify_element_gradient)
            energy_densities = energy_densities.at[elemIds].set(blockEnergyDensities)
            stresses = stresses.at[elemIds].set(blockStresses)
        return energy_densities, stresses


    def compute_initial_state():
        return _compute_initial_state_multi_block(fs, materialModels)

    
    return MechanicsFunctions(compute_strain_energy, jit(compute_updated_internal_variables), jit(compute_element_stiffnesses), jit(compute_output_energy_densities_and_stresses), compute_initial_state)


######
    
    
def create_mechanics_functions(functionSpace, mode2D, materialModel, pressureProjectionDegree=None):
    fs = functionSpace

    if mode2D == 'plane strain':
        grad_2D_to_3D = plane_strain_gradient_transformation
    elif mode2D == 'axisymmetric':
        grad_2D_to_3D = axisymmetric_element_gradient_transformation
    else:
        raise

    modify_element_gradient = grad_2D_to_3D
    if pressureProjectionDegree is not None:
        masterJ = Interpolants.make_master_tri_element(degree=pressureProjectionDegree)
        xigauss = functionSpace.quadratureRule.xigauss
        shapesJ = Interpolants.compute_shapes_on_tri(masterJ, xigauss)

        def modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
            elemGrads = volume_average_J_gradient_transformation(elemGrads, elemVols, shapesJ)
            return grad_2D_to_3D(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords)
    
    
    def compute_strain_energy(U, stateVariables):
        return _compute_strain_energy(fs, U, stateVariables, materialModel.compute_energy_density, modify_element_gradient)

        
    def compute_updated_internal_variables(U, stateVariables):
        return _compute_updated_internal_variables(fs, U, stateVariables, materialModel.compute_state_new, modify_element_gradient)

    
    def compute_element_stiffnesses(U, stateVariables):
        return _compute_element_stiffnesses(U, stateVariables, fs, materialModel.compute_energy_density, modify_element_gradient)


    output_lagrangian = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
    output_constitutive = value_and_grad(output_lagrangian, 1)

    
    def compute_output_energy_densities_and_stresses(U, stateVariables):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, output_constitutive, slice(None), modify_element_gradient=modify_element_gradient)

    
    def compute_initial_state():
        shape = Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 1
        return np.tile(materialModel.compute_initial_state(), shape)

    return MechanicsFunctions(compute_strain_energy, jit(compute_updated_internal_variables), jit(compute_element_stiffnesses), jit(compute_output_energy_densities_and_stresses), compute_initial_state)


def _compute_kinetic_energy(functionSpace, V, internals, density):
    def lagrangian_density(U, gradU, Q, X):
        return kinetic_energy_density(U, density)
    return FunctionSpace.integrate_over_block(functionSpace, V, internals, lagrangian_density, slice(None))


def _compute_element_masses(functionSpace, U, internals, density, modify_element_gradient):
    def lagrangian_density(V, gradV, Q, X):
        return kinetic_energy_density(V, density)
    f = vmap(compute_element_stiffness_from_global_fields, (None, None, 0 ,0 ,0 ,0 ,0, None, None))
    fs = functionSpace
    return f(U, fs.mesh.coords, internals, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols,
             lagrangian_density, modify_element_gradient)


def kinetic_energy_density(V, density):
    return 0.5*density*np.dot(V, V)


def compute_newmark_lagrangian(functionSpace, U, UPredicted, internals, density, dt, newmarkBeta, strain_energy_density, modify_element_gradient):
    # We can't quite fuse these kernels because KE uses the velocity field and
    # the strain energy uses the displacements. If profiling suggests fusing
    # is beneficial, we could add the time derivative field to the Lagrangian
    # density definition.

    def lagrangian_density(W, gradW, Q, X):
        return kinetic_energy_density(W, density)
    KE = FunctionSpace.integrate_over_block(functionSpace, U - UPredicted, internals, lagrangian_density,
                                            slice(None)) / (newmarkBeta*dt**2)

    lagrangian_density = strain_energy_density_to_lagrangian_density(strain_energy_density)
    SE = FunctionSpace.integrate_over_block(functionSpace, U, internals, lagrangian_density,
                                            slice(None), modify_element_gradient=modify_element_gradient)
    return SE + KE


def _compute_newmark_element_hessians(functionSpace, U, UPredicted, internals, density, dt, newmarkBeta, strain_energy_density, modify_element_gradient):
    def lagrangian_density(W, gradW, Q, X):
        return kinetic_energy_density(W, density)/(newmarkBeta*dt**2) + strain_energy_density(gradW, Q)
    f =  vmap(compute_element_stiffness_from_global_fields,
              (None, None, 0, 0, 0, 0, 0, None, None))
    fs = functionSpace
    UAlgorithmic = U - UPredicted
    return f(UAlgorithmic, fs.mesh.coords, internals, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols,
             lagrangian_density, modify_element_gradient)


def parse_2D_to_3D_gradient_transformation(mode2D):
    if mode2D == 'plane strain':
        grad_2D_to_3D = plane_strain_gradient_transformation
    elif mode2D == 'axisymmetric':
        grad_2D_to_3D = axisymmetric_element_gradient_transformation
    else:
        raise ValueError("Unrecognized value for mode2D")
    
    return grad_2D_to_3D


def define_pressure_projection_gradient_tranformation(pressureProjectionDegree, modify_element_gradient):
    if pressureProjectionDegree is not None:
        masterJ = Interpolants.make_master_tri_element(degree=pressureProjectionDegree)
        xigauss = functionSpace.quadratureRule.xigauss
        shapesJ = Interpolants.compute_shapes_on_tri(masterJ, xigauss)

        def modify_element_gradient_with_pressure_projection(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
            elemGrads = volume_average_J_gradient_transformation(elemGrads, elemVols, shapesJ)
            return modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords)

        modify_element_gradient = modify_element_gradient_with_pressure_projection
        
    return modify_element_gradient


def create_dynamics_functions(functionSpace, mode2D, materialModel, newmarkParameters, pressureProjectionDegree=None):
    fs = functionSpace

    modify_element_gradient = parse_2D_to_3D_gradient_transformation(mode2D)
    modify_element_gradient = define_pressure_projection_gradient_tranformation(
        pressureProjectionDegree, modify_element_gradient)

    def compute_algorithmic_energy(U, UPredicted, stateVariables, dt):
        return compute_newmark_lagrangian(functionSpace, U, UPredicted, stateVariables,
                                          materialModel.density, dt, newmarkParameters.beta,
                                          materialModel.compute_energy_density,
                                          modify_element_gradient)
    
    def compute_updated_internal_variables(U, stateVariables):
        return _compute_updated_internal_variables(fs, U, stateVariables, materialModel.compute_state_new, modify_element_gradient)
    
    def compute_element_hessians(U, UPredicted, stateVariables, dt):
        return _compute_newmark_element_hessians(
            functionSpace, U, UPredicted, stateVariables, materialModel.density, dt, 
            newmarkParameters.beta, materialModel.compute_energy_density, modify_element_gradient)

    output_lagrangian = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
    output_constitutive = value_and_grad(output_lagrangian, 1)
    def compute_output_potential_densities_and_stresses(U, stateVariables):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, output_constitutive, slice(None), modify_element_gradient=modify_element_gradient)

    def compute_kinetic_energy(V):
        stateVariables = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule)))
        return _compute_kinetic_energy(functionSpace, V, stateVariables, materialModel.density)

    def compute_output_strain_energy(U, stateVariables):
        return _compute_strain_energy(functionSpace, U, stateVariables, materialModel.compute_energy_density, modify_element_gradient)

    def compute_initial_state():
        shape = Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 1
        return np.tile(materialModel.compute_initial_state(), shape)

    def compute_element_masses():
        V = np.zeros_like(fs.mesh.coords)
        stateVariables = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule)))
        return _compute_element_masses(functionSpace, V, stateVariables, materialModel.density, modify_element_gradient)

    def predict(U, V, A, dt):
        U += dt*V + 0.5*dt*dt*(1.0 - 2.0*newmarkParameters.beta)*A
        V += dt*(1.0 - newmarkParameters.gamma)*A
        return U, V

    def correct(UCorrection, V, A, dt):
        A = UCorrection/(newmarkParameters.beta*dt*dt)
        V += dt*newmarkParameters.gamma*A
        return V, A

    return DynamicsFunctions(compute_algorithmic_energy,
                             jit(compute_updated_internal_variables),
                             jit(compute_element_hessians),
                             jit(compute_output_potential_densities_and_stresses),
                             jit(compute_kinetic_energy),
                             jit(compute_output_strain_energy),
                             compute_initial_state,
                             jit(compute_element_masses),
                             jit(predict),
                             jit(correct))
