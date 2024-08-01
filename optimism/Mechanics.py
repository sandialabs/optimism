from collections import namedtuple
from optimism.JaxConfig import *
from optimism import SparseMatrixAssembler
from optimism import FunctionSpace
from optimism import Mesh
from optimism.TensorMath import tensor_2D_to_3D
from optimism import QuadratureRule
from optimism import Interpolants
from typing import Callable, Dict, List, Optional
import equinox as eqx
import functools
import jax

# TODO add types to Callable's for improved autodocing
# TODO once all the equinox stuff is hooked up reconsider jitting by default here
class MechanicsFunctions(eqx.Module):
    fspace: FunctionSpace.FunctionSpace
    mat_models: List[any] # TODO type this
    modify_element_gradient: Callable
    element_hess_func: Callable
    L: Callable
    output_constitutive: Callable

    def __init__(self, fspace, mat_models, modify_element_gradient):
        self.fspace = fspace
        self.mat_models = mat_models
        self.modify_element_gradient = modify_element_gradient
        self.element_hess_func = hessian(FunctionSpace.integrate_element_from_local_field)
        self.L = strain_energy_density_to_lagrangian_density(mat_models[0].compute_energy_density)
        self.output_constitutive = jax.value_and_grad(self.L, 1)

    # public methods
    @eqx.filter_jit
    def compute_element_stiffnesses(self, U, state, dt=0.0):
        return self._compute_element_stiffnesses(U, state, dt)

    def compute_initial_state(self):
        shape = self.fspace.mesh.num_elements, QuadratureRule.len(self.fspace.quadratureRule), 1
        return np.tile(self.mat_models[0].compute_initial_state(), shape)

    @eqx.filter_jit
    def compute_output_energy_densities_and_stresses(self, U, state, dt=0.0):
        return FunctionSpace.evaluate_on_block(fs, U, state, dt, self.output_constitutive, slice(None), modify_element_gradient=modify_element_gradient)

    def compute_strain_energy(self, U, state, dt=0.0):
        L = strain_energy_density_to_lagrangian_density(self.mat_models[0].compute_energy_density)
        return FunctionSpace.integrate_over_block(
            self.fspace, U, state, dt, L,
            slice(None),
            modify_element_gradient=self.modify_element_gradient
        )

    @eqx.filter_jit
    def compute_updated_internal_variables(self, U, state, dt=0.0):
        dispGrads = FunctionSpace.compute_field_gradient(self.fspace, U, self.modify_element_gradient)
        dgQuadPointRavel = dispGrads.reshape(dispGrads.shape[0]*dispGrads.shape[1],*dispGrads.shape[2:])
        stQuadPointRavel = state.reshape(state.shape[0]*state.shape[1],*state.shape[2:])
        statesNew = vmap(self.mat_models[0].compute_state_new, (0, 0, None))(dgQuadPointRavel, stQuadPointRavel, dt)
        return statesNew.reshape(state.shape)

    @eqx.filter_jit
    def compute_output_material_qoi(self, U, state, dt=0.0):
        return FunctionSpace.evaluate_on_block(
            self.fspace, U, state, dt, 
            self._lagrangian_qoi, 
            slice(None), 
            modify_element_gradient=self.modify_element_gradient
        )

    def integrated_material_qoi(self, U, state, dt=0.0):
        return FunctionSpace.integrate_over_block(
            self.fspace, U, state, dt, 
            self._lagrangian_qoi,
            slice(None),
            modify_element_gradient=self.modify_element_gradient
        )

    # private methods
    # TODO there's probably some cleanup that can be done
    def _compute_element_stiffnesses(self, U, state, dt):
        f =  jax.vmap(self._compute_element_stiffness_from_global_fields,
                (None, None, 0, None, 0, 0, 0, 0, None))

                # (None, None, 0, None, 0, 0, 0, 0, None, None))
        fs = self.fspace
        return f(U, fs.mesh.coords, state, dt, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols, self.L)#,
                # self.L, self.modify_element_gradient)

    def _compute_element_stiffness_from_global_fields(self, U, coords, elInternals, dt, elConn, elShapes, elShapeGrads, elVols, func):#, lagrangian_density, modify_element_gradient):
        elDisp = U[elConn,:]
        elCoords = coords[elConn,:]
        return self.element_hess_func(elDisp, elCoords, elInternals, dt, elShapes, elShapeGrads,
                                elVols, func, self.modify_element_gradient)

    def _lagrangian_qoi(self, U, gradU, Q, X, dt):
        return self.mat_models[0].compute_material_qoi(gradU, Q, dt)


# TODO eventually deprecate this and make multi blocks the default
class MechanicsFunctionsMultiBlock(eqx.Module):
    fspace: FunctionSpace.FunctionSpace
    mat_models: List[any] # TODO type this
    modify_element_gradient: Callable
    element_hess_func: Callable
    Ls: Dict[str, Callable]
    output_constitutives: Dict[str, Callable]

    def __init__(self, fspace, mat_models, modify_element_gradient):
        self.fspace = fspace
        self.mat_models = mat_models
        self.modify_element_gradient = modify_element_gradient
        self.element_hess_func = hessian(FunctionSpace.integrate_element_from_local_field)
        self.Ls = {}
        self.output_constitutives = {}
        for key, val in mat_models.items():
            L = strain_energy_density_to_lagrangian_density(val.compute_energy_density)
            self.Ls[key] = L
            self.output_constitutives[key] = jax.value_and_grad(L, 1)

    # public methods
    @eqx.filter_jit
    def compute_element_stiffnesses(self, U, state, dt=0.0):
        fs, blockModels = self.fspace, self.mat_models
        # fs = functionSpace
        nen = Interpolants.num_nodes(fs.mesh.parentElement)
        elementHessians = np.zeros((fs.mesh.num_elements, nen, 2, nen, 2))

        for blockKey in blockModels:
            materialModel = blockModels[blockKey]
            L = strain_energy_density_to_lagrangian_density(materialModel.compute_energy_density)
            elemIds = fs.mesh.blocks[blockKey]
            f =  vmap(self._compute_element_stiffness_from_global_fields,
                    (None, None, 0, None, 0, 0, 0, 0, None))
            blockHessians = f(U, fs.mesh.coords, state[elemIds], dt, fs.mesh.conns[elemIds],
                            fs.shapes[elemIds], fs.shapeGrads[elemIds], fs.vols[elemIds],
                            L)
            elementHessians = elementHessians.at[elemIds].set(blockHessians)
        return elementHessians

    def compute_initial_state(self):
        fs, blockModels = self.fspace, self.mat_models

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

        initialState = np.zeros((fs.mesh.num_elements, numQuadPoints, numStateVariables))
        
        for blockKey in blockModels:
            elemIds = fs.mesh.blocks[blockKey]
            state = blockModels[blockKey].compute_initial_state()
            blockInitialState = np.tile(state, (elemIds.size, numQuadPoints, 1))
            initialState = initialState.at[elemIds, :, :blockInitialState.shape[2]].set(blockInitialState)

        return initialState

    @eqx.filter_jit
    def compute_output_energy_densities_and_stresses(self, U, state, dt=0.0):
        fs = self.fspace
        materialModels = self.mat_models
        energy_densities = np.zeros((fs.mesh.num_elements, QuadratureRule.len(fs.quadratureRule)))
        stresses = np.zeros((fs.mesh.num_elements, QuadratureRule.len(fs.quadratureRule), 3, 3))
        for blockKey in materialModels:
            compute_output_energy_density = materialModels[blockKey].compute_energy_density
            output_lagrangian = strain_energy_density_to_lagrangian_density(compute_output_energy_density)
            output_constitutive = value_and_grad(output_lagrangian, 1)
            elemIds = fs.mesh.blocks[blockKey]
            blockEnergyDensities, blockStresses = FunctionSpace.evaluate_on_block(fs, U, stateVariables, dt, output_constitutive, elemIds, modify_element_gradient=self.modify_element_gradient)
            energy_densities = energy_densities.at[elemIds].set(blockEnergyDensities)
            stresses = stresses.at[elemIds].set(blockStresses)
        return energy_densities, stresses

    def compute_strain_energy(self, U, state, dt=0.0):
        functionSpace, blockModels = self.fspace, self.mat_models
        energy = 0.0
        for blockKey in blockModels:
            materialModel = blockModels[blockKey]
            elemIds = functionSpace.mesh.blocks[blockKey]
            blockEnergy = FunctionSpace.integrate_over_block(functionSpace, U, state, dt, self.Ls[blockKey],
                                                             elemIds, modify_element_gradient=self.modify_element_gradient)
            
            energy += blockEnergy
        return energy

    @eqx.filter_jit
    def compute_updated_internal_variables(self, U, states, dt=0.0):
        functionSpace, blockModels = self.fspace, self.mat_models
        dispGrads = FunctionSpace.compute_field_gradient(functionSpace, U, self.modify_element_gradient)

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

    # @eqx.filter_jit
    def compute_output_material_qoi(self, U, state, dt=0.0):
        pass

    def integrated_material_qoi(self, U, state, dt=0.0):
        pass

    # private methods
    # TODO recycled from MechanicsFUnctions above, need to figure out how to override
    def _compute_element_stiffness_from_global_fields(self, U, coords, elInternals, dt, elConn, elShapes, elShapeGrads, elVols, func):#, lagrangian_density, modify_element_gradient):
        elDisp = U[elConn,:]
        elCoords = coords[elConn,:]
        return self.element_hess_func(elDisp, elCoords, elInternals, dt, elShapes, elShapeGrads,
                                elVols, func, self.modify_element_gradient)


# TODO sprinkle in some filter_jits
# TODO we can clean up some of those private methods below
class DynamicsFunctions(MechanicsFunctions):
    fspace: FunctionSpace.FunctionSpace
    mat_models: List[any] # TODO type this
    modify_element_gradient: Callable
    element_hess_func: Callable
    L: Callable
    output_constitutive: Callable
    newmark_parameters: any # get type right

    def __init__(
        self, 
        fspace, mat_models, modify_element_gradient, newmark_parameters,
    ):
        self.fspace = fspace
        self.mat_models = mat_models
        self.modify_element_gradient = modify_element_gradient
        self.newmark_parameters = newmark_parameters
        #
        self.element_hess_func = hessian(FunctionSpace.integrate_element_from_local_field)
        self.L = strain_energy_density_to_lagrangian_density(mat_models[0].compute_energy_density)
        self.output_constitutive = jax.value_and_grad(self.L, 1)

    # public methods
    def compute_algorithmic_energy(self, U, UPredicted, stateVariables, dt):
        return self.compute_newmark_lagrangian(U, UPredicted, stateVariables,
                                          self.mat_models[0].density, dt, self.newmark_parameters.beta)
    
    def compute_element_hessians(self, U, UPredicted, stateVariables, dt):
        return self._compute_newmark_element_hessians(
            U, UPredicted, stateVariables, self.mat_models[0].density, dt, 
            self.newmark_parameters.beta, self.mat_models[0].compute_energy_density, self.modify_element_gradient
        )

    def compute_element_masses(self):
        V = np.zeros_like(self.fspace.mesh.coords)
        stateVariables = np.zeros((self.fspace.mesh.num_elements, QuadratureRule.len(self.fspace.quadratureRule)))
        return self._compute_element_masses(V, stateVariables, self.mat_models[0].density, self.modify_element_gradient)

    def compute_newmark_lagrangian(self, U, UPredicted, internals, density, dt, newmarkBeta):
        # We can't quite fuse these kernels because KE uses the velocity field and
        # the strain energy uses the displacements. If profiling suggests fusing
        # is beneficial, we could add the time derivative field to the Lagrangian
        # density definition.

        def lagrangian_density(W, gradW, Q, X, dtime):
            return self._kinetic_energy_density(W, density)
        KE =  FunctionSpace.integrate_over_block(self.fspace, U - UPredicted, internals, dt,
                                                lagrangian_density, slice(None))
        KE *= 1 / (newmarkBeta*dt**2)

        lagrangian_density = strain_energy_density_to_lagrangian_density(self.mat_models[0].compute_energy_density)
        SE = FunctionSpace.integrate_over_block(self.fspace, U, internals, dt, lagrangian_density,
                                                slice(None), modify_element_gradient=self.modify_element_gradient)
        return SE + KE

    def compute_output_kinetic_energy(self, V):
        stateVariables = np.zeros((self.fspace.mesh.num_elements, QuadratureRule.len(self.fspace.quadratureRule)))
        return self._compute_kinetic_energy(V, stateVariables, self.mat_models[0].density)

    def compute_output_potential_densities_and_stresses(self, U, stateVariables, dt):
        return FunctionSpace.evaluate_on_block(self.fspace, U, stateVariables, dt, self.output_constitutive, slice(None), modify_element_gradient=self.modify_element_gradient)

    def compute_output_strain_energy(self, U, stateVariables, dt):
        return self.compute_strain_energy(U, stateVariables, dt)

    # newmark methods
    def correct(self, UCorrection, V, A, dt):
        A = UCorrection/(self.newmark_parameters.beta*dt*dt)
        V += dt*self.newmark_parameters.gamma*A
        return V, A

    def predict(self, U, V, A, dt):
        U += dt*V + 0.5*dt*dt*(1.0 - 2.0*self.newmark_parameters.beta)*A
        V += dt*(1.0 - self.newmark_parameters.gamma)*A
        return U, V

    # private methods
    def _compute_kinetic_energy(self, V, internals, density):
        def lagrangian_density(U, gradU, Q, X, dt):
            return self._kinetic_energy_density(U, density)
        unused = 0.0
        return FunctionSpace.integrate_over_block(self.fspace, V, internals, unused, lagrangian_density, slice(None))

    def _compute_element_masses(self, U, internals, density):
        def lagrangian_density(V, gradV, Q, X, dt):
            return self._kinetic_energy_density(V, density)
        f = vmap(self._compute_element_stiffness_from_global_fields, (None, None, 0, None, 0 ,0 ,0 ,0, None))
        fs = self.fspace
        unusedDt = 0.0
        return f(U, fs.mesh.coords, internals, unusedDt, fs.mesh.conns, fs.shapes, fs.shapeGrads,
                fs.vols, lagrangian_density)

    def _compute_newmark_element_hessians(self, U, UPredicted, internals, density, dt, newmarkBeta, strain_energy_density, modify_element_gradient):
        def lagrangian_density(W, gradW, Q, X, dtime):
            return self._kinetic_energy_density(W, density)/(newmarkBeta*dtime**2) + strain_energy_density(gradW, Q, dtime)
        f =  vmap(self._compute_element_stiffness_from_global_fields,
                (None, None, 0, None, 0, 0, 0, 0, None)
        )
        fs = self.fspace
        UAlgorithmic = U - UPredicted
        return f(UAlgorithmic, fs.mesh.coords, internals, dt, fs.mesh.conns, fs.shapes, fs.shapeGrads, fs.vols,
                lagrangian_density)

    def _kinetic_energy_density(self, V, density):
        # assumes spatially homogeneous density
        return 0.5*density*np.dot(V, V)

# NewmarkParameters = namedtuple('NewmarkParameters', ['gamma', 'beta'], defaults=[0.5, 0.25])
class NewmarkParameters(eqx.Module):
    gamma: Optional[float] = 0.5
    beta: Optional[float] = 0.25

# gradient transformations
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


###### constructors 
def create_dynamics_functions(functionSpace, mode2D, materialModel, newmarkParameters, pressureProjectionDegree=None):
    fs = functionSpace

    modify_element_gradient = parse_2D_to_3D_gradient_transformation(mode2D)
    modify_element_gradient = define_pressure_projection_gradient_tranformation(
        pressureProjectionDegree, modify_element_gradient)

    return DynamicsFunctions(
        functionSpace, [materialModel], modify_element_gradient, newmarkParameters,
    )
 
    
# TODO remove this
def create_mechanics_functions_old(functionSpace, mode2D, materialModel, 
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

    return MechanicsFunctions(functionSpace, [materialModel], modify_element_gradient)

# TODO deprecate this eventually after fixing tests and examples
def create_mechanics_functions(functionSpace, mode2D, materialModel, pressureProjectionDegree=None):
    # just make it a multi-block problem but make the name for the mat model
    # the same name as the only block in the mesh
    print(functionSpace.mesh.blocks)
    print(functionSpace.mesh)
    materialModels = {
        list(functionSpace.mesh.blocks.keys())[0]: materialModel
    }
    return create_multi_block_mechanics_functions(
        functionSpace, mode2D, materialModels, 
        pressureProjectionDegree=pressureProjectionDegree
    )


def create_multi_block_mechanics_functions(functionSpace, mode2D, materialModels, pressureProjectionDegree=None):
    fs = functionSpace

    if mode2D == 'plane strain':
        grad_2D_to_3D = plane_strain_gradient_transformation
    elif mode2D == 'axisymmetric':
        grad_2D_to_3D = axisymmetric_element_gradient_transformation
    else:
        raise NotImplementedError

    modify_element_gradient = grad_2D_to_3D
    if pressureProjectionDegree is not None:
        masterJ = Interpolants.make_master_tri_element(degree=pressureProjectionDegree)
        xigauss = functionSpace.quadratureRule.xigauss
        shapesJ = Interpolants.compute_shapes_on_tri(masterJ, xigauss)

        def modify_element_gradient(elemGrads, elemVols):
            elemGrads = volume_average_J_gradient_transformation(elemGrads, elemVols, shapesJ)
            return grad_2D_to_3D(elemGrads, elemVols)
    
    return MechanicsFunctionsMultiBlock(functionSpace, materialModels, modify_element_gradient)


# oddballs, where should y'all go?
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


def strain_energy_density_to_lagrangian_density(strain_energy_density):
    def L(U, gradU, Q, X, dt):
        return strain_energy_density(gradU, Q, dt)
    return L
