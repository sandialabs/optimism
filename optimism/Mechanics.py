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


def sparse_matrix_from_energy_density(U, internals, functionSpace, dofManager, compute_energy_density):
     elementHessians = compute_element_stiffnesses(U, internals, functionSpace, compute_energy_density)
     K = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementHessians,
                                                                functionSpace.conns,
                                                                dofManager)
     return K


def _compute_strain_energy(functionSpace, UField, stateField,
                           compute_energy_density,
                           modify_element_gradient):
    L = strain_energy_density_to_lagrangian_density(compute_energy_density)
    return FunctionSpace.integrate_over_block(functionSpace, UField, stateField, L,
                                              slice(None),
                                              modify_element_gradient)


def _compute_strain_energy_multi_block(functionSpace, UField, stateField, blockModels,
                                       modify_element_gradient):
    energy = 0.0
    for blockKey in blockModels:
        materialModel = blockModels[blockKey]
        elemIds = functionSpace.mesh.blocks[blockKey]
        
        L = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
        
        blockEnergy = FunctionSpace.integrate_over_block(functionSpace, UField, stateField, L,
                                                         elemIds, modify_element_gradient)
        
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
        blockInitialState = blockModels[blockKey].compute_initial_state( (elemIds.size, numQuadPoints, 1) )
        initialState = initialState.at[elemIds, :, :blockInitialState.shape[2]].set(blockInitialState)

    return initialState


def _compute_element_stiffnesses_multi_block(U, stateVariables, functionSpace, blockModels, modify_element_gradient):
    fs = functionSpace
    nen = Interpolants.num_nodes(functionSpace.mesh.masterElement)
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
            compute_output_energy_density = materialModels[blockKey].compute_output_energy_density
            output_lagrangian = strain_energy_density_to_lagrangian_density(compute_output_energy_density)
            output_constitutive = value_and_grad(output_lagrangian, 1)
            elemIds = fs.mesh.blocks[blockKey]
            blockEnergyDensities, blockStresses = FunctionSpace.evaluate_on_block(fs, U, stateVariables, output_constitutive, elemIds, modify_element_gradient)
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

    
    compute_output_energy_density = materialModel.compute_output_energy_density

    output_lagrangian = strain_energy_density_to_lagrangian_density(compute_output_energy_density)
    output_constitutive = value_and_grad(output_lagrangian, 1)

    
    def compute_output_energy_densities_and_stresses(U, stateVariables):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, output_constitutive, slice(None), modify_element_gradient)

    
    def compute_initial_state():
        return materialModel.compute_initial_state((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 1))

    return MechanicsFunctions(compute_strain_energy, jit(compute_updated_internal_variables), jit(compute_element_stiffnesses), jit(compute_output_energy_densities_and_stresses), compute_initial_state)


def compute_element_mass_matrix(density, elemShapes, elemVols):
    m = elemShapes.T@np.diag(density*elemVols)@elemShapes
    M = np.zeros((m.shape[0],2,m.shape[0],2))
    for i in range(2):
        M = M.at[:,i,:,i].set(m)
    return M

    
def compute_element_masses(density, mesh):
    elementOrder = mesh.masterElement.degree
    quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(elementOrder))
    fs = FunctionSpace.construct_function_space(mesh, quadRule)
    return vmap(compute_element_mass_matrix, (None,0,0))(density, fs.shapes, fs.vols)


def compute_element_kinetic_energy(V, elementMass, elementConn):
    elementVs = V[elementConn,:].ravel()
    rowSz = elementMass.shape[0]*elementMass.shape[1]
    return 0.5*elementVs@elementMass.reshape(rowSz,rowSz)@elementVs


def compute_kinetic_energy(V, elementMasses, elementConns):
    def f(carry, x):
        elemConn, elemMass = x
        elemT = compute_element_kinetic_energy(V, elemMass, elemConn)
        carry += elemT
        return carry, elemT
    
    T, elemKineticEnergies = lax.scan(f, 0.0, (elementConns,elementMasses))
    return T


def parse_2D_to_3D_gradient_transformation(mode2D):
    if mode2D == 'plane strain':
        grad_2D_to_3D = plane_strain_gradient_transformation
    elif mode2D == 'axisymmetric':
        grad_2D_to_3D = axisymmetric_element_gradient_transformation
    else:
        raise ValueError("Unrecognized value for mode2D")
    
    return grad_2D_to_3D


def compute_algorithmic_kinetic_energy(U, UPredicted, elementMasses, elementConns,
                                       dt, newmarkBeta):
    V = (U - UPredicted)/dt
    return compute_kinetic_energy(V, elementMasses, elementConns)/newmarkBeta


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


def create_dynamics_functions(functionSpace, mode2D, materialModel, newmarkParameters, elementMasses, pressureProjectionDegree=None):
    fs = functionSpace

    modify_element_gradient = parse_2D_to_3D_gradient_transformation(mode2D)
    modify_element_gradient = define_pressure_projection_gradient_tranformation(
        pressureProjectionDegree, modify_element_gradient)
  
    def compute_algorithmic_energy(U, UPredicted, stateVariables, dt):
        PE = _compute_strain_energy(fs, U, stateVariables, materialModel.compute_energy_density, modify_element_gradient)
        KE = compute_algorithmic_kinetic_energy(U, UPredicted, elementMasses, fs.mesh.conns,
                                                dt, newmarkParameters.beta)
        return PE + KE
    
    def compute_updated_internal_variables(U, stateVariables):
        return _compute_updated_internal_variables(fs, U, stateVariables, materialModel.compute_state_new, modify_element_gradient)
    
    def compute_element_hessians(U, stateVariables, dt):
        elementStiffnesses = _compute_element_stiffnesses(U, stateVariables, fs, materialModel.compute_energy_density, modify_element_gradient)
        return elementStiffnesses + elementMasses/(newmarkParameters.beta*dt*dt)

    def compute_output_kinetic_energy(V):
        return compute_kinetic_energy(V, elementMasses, fs.mesh.conns)

    def compute_output_strain_energy(U, stateVariables):
        return _compute_strain_energy(functionSpace, U, stateVariables, materialModel.compute_energy_density, modify_element_gradient)

    compute_output_energy_density = materialModel.compute_output_energy_density
    output_lagrangian = strain_energy_density_to_lagrangian_density(compute_output_energy_density)
    output_constitutive = value_and_grad(output_lagrangian, 1)
    def compute_output_potential_densities_and_stresses(U, stateVariables):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, output_constitutive, slice(None), modify_element_gradient)

    def compute_initial_state():
        return materialModel.compute_initial_state((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 1))

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
                             jit(compute_output_kinetic_energy),
                             jit(compute_output_strain_energy),
                             compute_initial_state,
                             jit(predict),
                             jit(correct))
