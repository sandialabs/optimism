from collections import namedtuple

from optimism.JaxConfig import *
from optimism import SparseMatrixAssembler
from optimism import FunctionSpace
from optimism import Mesh
from optimism.TensorMath import tensor_2D_to_3D
from optimism import QuadratureRule
from optimism import Interpolants
import jax

MechanicsFunctions = namedtuple('MechanicsFunctions',
                                ['compute_strain_energy',
                                 'compute_updated_internal_variables',
                                 'compute_element_stiffnesses',
                                 'compute_output_energy_densities_and_stresses',
                                 'compute_initial_state',
                                 'integrated_material_qoi',
                                 'compute_output_material_qoi'])


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


def compute_element_stiffness_from_global_fields(U, coords, elInternals, props, dt, elConn, elShapes, elShapeGrads, elVols, lagrangian_density, modify_element_gradient):
    elDisp = U[elConn,:]
    elCoords = coords[elConn,:]
    return element_hess_func(elDisp, elCoords, elInternals, props, dt, elShapes, elShapeGrads,
                             elVols, lagrangian_density, modify_element_gradient)


# TODO once we map thingst o equinox classes, make these methods bound to the class
# TODO use jax.lax.cond below to make this jit safe
# aprently this is unjittable
def vmapPropValue(propArray):
    numAxes = len(propArray.shape)
    if numAxes > 1:
        return 0
    else:
        return None
    # return 0
    # jit safe
    # return jax.lax.cond(numAxes > 1, lambda _: (0,), lambda _: (None,), numAxes) # do I need the numAxes on the end?


def fixed_props_to_element_props(props, num_el):
    num_axes = len(props.shape)
    # below is not jittable either
    # new_props = jax.lax.cond(
    #     num_axes > 1, 
    #     lambda x: (x,),
    #     lambda x: (np.repeat(x.reshape((-1, 1)), num_el, axis=1),),
    #     props
    # )
    if num_axes > 1:
        new_props = props
    else:
        new_props = np.repeat(props.reshape((-1, 1)), num_el, axis=1)
    return new_props


def tile_props(props, n_el, n_q):
    num_axes = len(props.shape)
    if num_axes > 1:
        tile_axes = (1, n_q, 1)
        new_props = np.tile(props, tile_axes)
        new_props = new_props.reshape((new_props.shape[0] * new_props.shape[1], new_props.shape[2]))
    else:
        new_props = props
    return new_props

# Note this is for the case where properties are constant across a block
def _compute_element_stiffnesses(U, internals, props, dt, functionSpace, compute_energy_density, modify_element_gradient):
    L = strain_energy_density_to_lagrangian_density(compute_energy_density)
    vmapValue = vmapPropValue(props)
    f =  vmap(compute_element_stiffness_from_global_fields,
              (None, None, 0, vmapValue, None, 0, 0, 0, 0, None, None))
    fs = functionSpace
    return f(U, fs.mesh.coords, internals, props, dt, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols,
             L, modify_element_gradient)


# TODO we need a _compute_element_stifnesses method for when properties are variable over a block
# Lucas this is where you can try out writing your own
# Note that the 'None' in the fourth slot should now be 0
# other than that I think everything is the same


# TODO we'll also need a method here that calls FunctionSpace.integrate_block
def _compute_strain_energy(functionSpace, UField, stateField, props, dt,
                           compute_energy_density,
                           modify_element_gradient):
    L = strain_energy_density_to_lagrangian_density(compute_energy_density)
    return FunctionSpace.integrate_over_block(functionSpace, UField, stateField, props, dt, L,
                                              slice(None),
                                              modify_element_gradient=modify_element_gradient)

# TODO add props
def _compute_strain_energy_multi_block(functionSpace, UField, stateField, dt, blockModels,
                                       modify_element_gradient):
    energy = 0.0
    for blockKey in blockModels:
        materialModel = blockModels[blockKey]
        elemIds = functionSpace.mesh.blocks[blockKey]
        
        L = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
        
        blockEnergy = FunctionSpace.integrate_over_block(functionSpace, UField, stateField, dt, L,
                                                         elemIds, modify_element_gradient=modify_element_gradient)
        
        energy += blockEnergy
    return energy


# TODO fix this, props wrong size
def _compute_updated_internal_variables(functionSpace, U, states, props, dt, compute_state_new, modify_element_gradient):
    # U -> (n_nodes, n_dims) -> Nodal field
    # state -> (n_els, n_quadrature_points, n_states) -> Quadrature field
    dispGrads = FunctionSpace.compute_field_gradient(functionSpace, U, modify_element_gradient)
    # dispGrads -> (n_els, n_quadrature_points, n_dims, n_dims) -> Quadrature field
    dgQuadPointRavel = dispGrads.reshape(dispGrads.shape[0]*dispGrads.shape[1],*dispGrads.shape[2:])
    # dgQuadPointRavel -> (n_els * n_quadrature_points, n_dims, n_dims) -> Quadrature field
    stQuadPointRavel = states.reshape(states.shape[0]*states.shape[1],*states.shape[2:])
    # new stuff below
    # really what we need to do is switch on whether props are already element based
    # so we have to check the sizes to see if it's (np,) which would be the case of
    # fixed properties for elements in teh block or (ne, np) which is the case for
    # element bound properties. Then based on this we either have 
    # repeat (np,) to be (ne, np) or do nothing and then
    # repeat so things are (ne, nq, np) then flatten to be (ne * nq, np)
    prop_vmap_axes = vmapPropValue(props) # -> 0 - vmap over all quadrature points for properties or None - don't vmap over quadrature points for properties
    new_props = tile_props(props, dispGrads.shape[0], dispGrads.shape[1]) # -> (n_els * n_quadrature_pts, n_props) or (n_props,)
    statesNew = vmap(compute_state_new, (0, 0, prop_vmap_axes, None))(dgQuadPointRavel, stQuadPointRavel, new_props, dt)
    # what lucas did below
    #    props_edited = np.transpose(props)
    #    props_edited_a = np.repeat(props_edited,3,axis=1)
    #    props_edited_b = np.reshape(props_edited_a,(np.shape(props)[1],np.shape(props)[0],3))
    #    statesNew = vmap(compute_state_new, (0, 0, vmapPropValue(props), None))(dgQuadPointRavel, stQuadPointRavel, props_edited_b, dt)
    return statesNew.reshape(states.shape)
    # return states

# TODO add props
def _compute_updated_internal_variables_multi_block(functionSpace, U, states, dt, blockModels, modify_element_gradient):
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
        blockStatesNew = vmap(compute_state_new, (0, 0, None))(dgQuadPointRavel, stQuadPointRavel, dt).reshape(blockStates.shape)
        statesNew = statesNew.at[elemIds, :, :blockStatesNew.shape[2]].set(blockStatesNew)
        

    return statesNew


# TODO add props
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


# TODO add props
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
    def L(U, gradU, Q, props, X, dt):
        return strain_energy_density(gradU, Q, props, dt)
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
    
    
    def compute_strain_energy(U, stateVariables, dt=0.0):
        return _compute_strain_energy_multi_block(fs, U, stateVariables, dt, materialModels, modify_element_gradient)

        
    def compute_updated_internal_variables(U, stateVariables, dt=0.0):
        return _compute_updated_internal_variables_multi_block(fs, U, stateVariables, dt, materialModels, modify_element_gradient)

    
    def compute_element_stiffnesses(U, stateVariables, dt=0.0):
        return _compute_element_stiffnesses_multi_block(U, stateVariables, functionSpace, materialModels, modify_element_gradient)


    def compute_output_energy_densities_and_stresses(U, stateVariables, dt=0.0):
        energy_densities = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule)))
        stresses = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 3, 3))
        for blockKey in materialModels:
            compute_output_energy_density = materialModels[blockKey].compute_energy_density
            output_lagrangian = strain_energy_density_to_lagrangian_density(compute_output_energy_density)
            output_constitutive = value_and_grad(output_lagrangian, 1)
            elemIds = fs.mesh.blocks[blockKey]
            blockEnergyDensities, blockStresses = FunctionSpace.evaluate_on_block(fs, U, stateVariables, dt, output_constitutive, elemIds, modify_element_gradient=modify_element_gradient)
            energy_densities = energy_densities.at[elemIds].set(blockEnergyDensities)
            stresses = stresses.at[elemIds].set(blockStresses)
        return energy_densities, stresses


    def compute_initial_state():
        return _compute_initial_state_multi_block(fs, materialModels)

    
    return MechanicsFunctions(compute_strain_energy, jit(compute_updated_internal_variables), jit(compute_element_stiffnesses), jit(compute_output_energy_densities_and_stresses), compute_initial_state, None, None)


######
    
    
def create_mechanics_functions(functionSpace, mode2D, materialModel, 
                               pressureProjectionDegree=None):
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
    
    
    def compute_strain_energy(U, stateVariables, props, dt=0.0):
        return _compute_strain_energy(fs, U, stateVariables, props, dt, materialModel.compute_energy_density, modify_element_gradient)

        
    # TODO add props
    def compute_updated_internal_variables(U, stateVariables, props, dt=0.0):
        return _compute_updated_internal_variables(fs, U, stateVariables, props, dt, materialModel.compute_state_new, modify_element_gradient)

    
    def compute_element_stiffnesses(U, stateVariables, props, dt=0.0):
        return _compute_element_stiffnesses(U, stateVariables, props, dt, fs, materialModel.compute_energy_density, modify_element_gradient)


    output_lagrangian = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
    output_constitutive = value_and_grad(output_lagrangian, 1)

    
    def compute_output_energy_densities_and_stresses(U, stateVariables, props, dt=0.0):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, props, dt, output_constitutive, slice(None), modify_element_gradient=modify_element_gradient)

    
    def compute_initial_state():
        shape = Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule), 1
        return np.tile(materialModel.compute_initial_state(), shape)

    def lagrangian_qoi(U, gradU, Q, props, X, dt):
        return materialModel.compute_material_qoi(gradU, Q, props, dt)

    def integrated_material_qoi(U, stateVariables, props, dt=0.0):

        return FunctionSpace.integrate_over_block(fs, U, stateVariables, props, dt, 
                                                  lagrangian_qoi,
                                                  slice(None),
                                                  modify_element_gradient=modify_element_gradient)

    def compute_output_material_qoi(U, stateVariables, props, dt=0.0):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, props, dt, lagrangian_qoi, slice(None), modify_element_gradient=modify_element_gradient)

    return MechanicsFunctions(compute_strain_energy, jit(compute_updated_internal_variables), jit(compute_element_stiffnesses), jit(compute_output_energy_densities_and_stresses), compute_initial_state, integrated_material_qoi, jit(compute_output_material_qoi))


# TODO need to update this for props. Eigen won't work otherwise
def _compute_kinetic_energy(functionSpace, V, internals, props, density):
    def lagrangian_density(U, gradU, Q, X, P, dt):
        return kinetic_energy_density(U, density)
    unused = 0.0
    return FunctionSpace.integrate_over_block(functionSpace, V, internals, props, unused, lagrangian_density, slice(None))

# TODO need to update this for props. Eigen won't work otherwise
def _compute_element_masses(functionSpace, U, internals, density, modify_element_gradient):
    def lagrangian_density(V, gradV, Q, X, dt):
        return kinetic_energy_density(V, density)
    f = vmap(compute_element_stiffness_from_global_fields, (None, None, 0, None, 0 ,0 ,0 ,0, None, None))
    fs = functionSpace
    unusedDt = 0.0
    return f(U, fs.mesh.coords, internals, unusedDt, fs.mesh.conns, fs.shapes, fs.shapeGrads,
             fs.vols, lagrangian_density, modify_element_gradient)


def kinetic_energy_density(V, density):
    # assumes spatially homogeneous density
    return 0.5*density*np.dot(V, V)


def compute_newmark_lagrangian(functionSpace, U, UPredicted, internals, density, dt, newmarkBeta, strain_energy_density, modify_element_gradient):
    # We can't quite fuse these kernels because KE uses the velocity field and
    # the strain energy uses the displacements. If profiling suggests fusing
    # is beneficial, we could add the time derivative field to the Lagrangian
    # density definition.

    def lagrangian_density(W, gradW, Q, X, dtime):
        return kinetic_energy_density(W, density)
    KE =  FunctionSpace.integrate_over_block(functionSpace, U - UPredicted, internals, dt,
                                             lagrangian_density, slice(None))
    KE *= 1 / (newmarkBeta*dt**2)

    lagrangian_density = strain_energy_density_to_lagrangian_density(strain_energy_density)
    SE = FunctionSpace.integrate_over_block(functionSpace, U, internals, dt, lagrangian_density,
                                            slice(None), modify_element_gradient=modify_element_gradient)
    return SE + KE


def _compute_newmark_element_hessians(functionSpace, U, UPredicted, internals, density, dt, newmarkBeta, strain_energy_density, modify_element_gradient):
    def lagrangian_density(W, gradW, Q, X, dtime):
        return kinetic_energy_density(W, density)/(newmarkBeta*dtime**2) + strain_energy_density(gradW, Q, dtime)
    f =  vmap(compute_element_stiffness_from_global_fields,
              (None, None, 0, None, 0, 0, 0, 0, None, None))
    fs = functionSpace
    UAlgorithmic = U - UPredicted
    return f(UAlgorithmic, fs.mesh.coords, internals, dt, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols,
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
    
    def compute_updated_internal_variables(U, stateVariables, dt):
        return _compute_updated_internal_variables(fs, U, stateVariables, dt, materialModel.compute_state_new, modify_element_gradient)
    
    def compute_element_hessians(U, UPredicted, stateVariables, dt):
        return _compute_newmark_element_hessians(
            functionSpace, U, UPredicted, stateVariables, materialModel.density, dt, 
            newmarkParameters.beta, materialModel.compute_energy_density, modify_element_gradient)

    output_lagrangian = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
    output_constitutive = value_and_grad(output_lagrangian, 1)
    def compute_output_potential_densities_and_stresses(U, stateVariables, dt):
        return FunctionSpace.evaluate_on_block(fs, U, stateVariables, dt, output_constitutive, slice(None), modify_element_gradient=modify_element_gradient)

    def compute_kinetic_energy(V,props):
        stateVariables = np.zeros((Mesh.num_elements(fs.mesh), QuadratureRule.len(fs.quadratureRule)))
        return _compute_kinetic_energy(functionSpace, V, stateVariables, props, materialModel.density)

    def compute_output_strain_energy(U, stateVariables, dt):
        return _compute_strain_energy(functionSpace, U, stateVariables, dt, materialModel.compute_energy_density, modify_element_gradient)

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


# TODO need to update this for props. Eigen won't work otherwise
def compute_traction_potential_energy(fs, U, quadRule, edges, load):
    """Compute potential energy of surface tractions.

    Arguments:
    fs: a FunctionSpace object
    U: the nodal displacements
    quadRule: the 1D quadrature rule to use for the integration
    edges: array of edges, each row is an edge. Each edge has two entries, the
         element ID, and the permutation of that edge in the triangle (0, 1,
         2).
    load: Callable that returns the traction vector. The signature is
        load(X, n), where X is coordinates of a material point, and n is the
        outward unit normal.
    time: current time
    """
    def compute_energy_density(u, X, n):
        traction = load(X, n)
        return -np.dot(u, traction)
    return FunctionSpace.integrate_function_on_edges(fs, compute_energy_density, U, quadRule, edges)
