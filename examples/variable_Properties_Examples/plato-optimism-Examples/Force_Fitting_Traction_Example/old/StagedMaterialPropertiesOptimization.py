from jax import grad
from jax import jit
from jax import vmap
from optimism import EquationSolver
from optimism import VTKWriter
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism.FunctionSpace import DofManager
from optimism.FunctionSpace import EssentialBC
from optimism.material import Neohookean_VariableProps
from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace
from scipy.sparse import linalg

import jax.numpy as np
import numpy as onp
from collections import namedtuple

EnergyFunctions = namedtuple('EnergyFunctions',
                            ['energy_function_props',
                             'nodal_forces'])

# simulation parameterized on material properties
class MaterialPropertiesOptimization:

    def __init__(self):
        self.targetSteps = [10, 20, 30, 40, 50, 60, 70, 80] 
        self.targetStepsDirty = [10, 20, 30, 40, 30, 20, 10, 0] # any step over the step per stage doesn't translate properly to the loading/unloading cycle
        self.targetForces = [-12, -4, -7, -30, -6, -4, -10, 0] # target plateau response
        self.scaleObjective = 1.0 # -1.0 to maximize
        self.stateNotStored = True
        self.writeOutput = True

        self.quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        self.ebcs = [
            EssentialBC(nodeSet='bottom_sideset', component=0),
            EssentialBC(nodeSet='bottom_sideset', component=1),
            EssentialBC(nodeSet='top_sideset', component=0),
            EssentialBC(nodeSet='top_sideset', component=1),
            EssentialBC(nodeSet='right_sideset',component=0),
            EssentialBC(nodeSet='left_sideset',component=0)
        ]

        # TODO: constant properties are hard-coded in material model for now
        constant_props = {
            'density': 1.0
        }
        self.mat_model = Neohookean_VariableProps.create_material_model_functions(constant_props, 'adagio')

        self.eq_settings = EquationSolver.get_settings(
            max_trust_iters=100,
            tr_size=0.25,
            min_tr_size=1e-15,
            tol=5e-8
        )

        self.input_mesh = './EXO_files/ellipse_test_Seeded.exo'
        origMesh = ReadExodusMesh.read_exodus_mesh(self.input_mesh)

        # first order mesh
        # nodeSets = Mesh.create_nodesets_from_sidesets(origMesh)
        # self.mesh = Mesh.mesh_with_nodesets(origMesh, nodeSets)

        # second order mesh
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(origMesh, order=2, createNodeSetsFromSideSets=True)
        self.index = (self.mesh.nodeSets['yplus_sideset'], 1)
        self.func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        self.dof_manager = DofManager(self.func_space, 2, self.ebcs)
        self.mech_funcs = Mechanics.create_mechanics_functions(self.func_space, mode2D='plane strain', materialModel=self.mat_model)

        self.plot_file = 'force_control_response.npz'

        self.maxForce = 40
        self.maxDisp = 0
        self.stages = 2
        self.steps_per_stage = 40
        self.steps = self.stages * self.steps_per_stage

    

    def create_field(self, Uu, disp):
        def get_ubcs(disp):
            V = np.zeros(self.mesh.coords.shape)
            V = V.at[self.index].set(disp)
            return self.dof_manager.get_bc_values(V)

        return self.dof_manager.create_field(Uu, get_ubcs(disp))

    def import_parameters(self, materialProperties=[]):
        if not materialProperties:
            raise ValueError('Material properties were not passed to MaterialPropertiesOptimization import_parameters function.')

        if len(self.mesh.blocks) > 1:
            raise ValueError('Global element ID mapping is currently only set up for single block.')
        
        self.elementMap = onp.argsort(self.mesh.block_maps['Block1'])

        print(self.elementMap)
        # print(materialProperties)        

        # TODO: convert properties from 0-1 to 0-255 (for now just do a linear map)
        # TODO: convert the RGB value to light intensity using the correct relations

        materialProperties = np.array(materialProperties)

        matPropConv = materialProperties

        props = matPropConv.at[self.elementMap].get()
        props = props.reshape((props.shape[0], 1))

        self.elementProperties = props

        self.stateNotStored = True
        self.state = []

    def run_simulation(self):
        # methods defined on the fly

        def energy_function(Uu, p):
            U = self.create_field(Uu)
            internal_variables = p.state_data
            strainEnergy = self.mech_funcs.compute_strain_energy(U, internal_variables, self.dt)

            F = p.bc_data
            def force_function(x, n):
                return np.array([0.0, F/self.tractionArea])

            loadPotential = Mechanics.compute_traction_potential_energy(
                self.func_space, U, self.lineQuadRule, self.mesh.sideSets['top_sideset'], 
                force_function)

            return strainEnergy + loadPotential

        # def energy_function_all_dofs(U, p):
        #     internal_variables = p.state_data
        #     strainEnergy = self.mech_funcs.compute_strain_energy(U, internal_variables, self.elementProperties)
        #     loadPotential = Mechanics.compute_traction_potential_energy(self.func_space, U, self.lineQuadRule, self.mesh.sideSets['top_sideset'],
        #                                                                 lambda x, n: np.array([0.0, -self.maxForce]))
        #     return self.mech_funcs.compute_strain_energy(U, internal_variables, self.elementProperties)
        

        # def energy_function(Uu, p):
        #     internal_variables = p.state_data
        #     U = self.create_field(Uu, p.bc_data)
        #     strainEnergy = self.mech_funcs.compute_strain_energy(U, internal_variables, self.elementProperties)
        #     loadPotential = Mechanics.compute_traction_potential_energy(self.func_space, U, self.lineQuadRule, self.mesh.sideSets['top_sideset'],
        #                                                                 lambda x, n: np.array([0.0, -self.maxForce]))


        #     return energy_function_all_dofs(U, p)

        # nodal_forces = jit(grad(energy_function_all_dofs, argnums=0))

        def assemble_sparse(Uu, p):
            U = self.create_field(Uu, p.bc_data)
            internal_variables = p.state_data
            element_stiffnesses = self.mech_funcs.compute_element_stiffnesses(U, internal_variables, self.elementProperties)
            return SparseMatrixAssembler.\
                assemble_sparse_stiffness_matrix(element_stiffnesses, self.func_space.mesh.conns, self.dof_manager)
    
        def store_force_displacement(Uu, forceval, force, disp):
            U = self.create_field(Uu, p.bc_data)
            # f = nodal_forces(U, p)
            # index = (self.mesh.nodeSets['top_sideset'],1)
            # force.append( onp.sum(onp.array(f.at[self.index].get())) )
            force.append( forceval )
            disp.append(np.mean(U.at[self.index].get()))
            # disp.append( dispval )

            with open(self.plot_file,'wb') as U:
                np.savez(U, force=force, displacement=disp,
                targetForces=np.array(self.targetForces), 
                targetDisplacements=(self.maxDisp/self.steps_per_stage) * np.array(self.targetStepsDirty))


        def write_vtk_output(Uu, p, step):
            U = self.create_field(Uu, p.bc_data)
            plotName = 'output-'+str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='displ', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

            energyDensities = self.mech_funcs.compute_output_energy_densities_and_stresses(U, p.state_data, self.elementProperties)[0]
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(self.func_space, energyDensities)
            writer.add_cell_field(name='strain_energy_density',
                                  cellData=cellEnergyDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.add_cell_field(name='element_property_field', 
                                  cellData=self.elementProperties, 
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.write()

        # problem set up
        Uu = self.dof_manager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = self.mech_funcs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs, prop_data=self.elementProperties)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        self.objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        # loop over load steps
        force = 0.
        disp = 0.
        fd_force = []
        fd_disp = []

        store_force_displacement(Uu, force, fd_force, fd_disp)
        self.state.append((Uu, p))

        globalStep = 1

        steps_per_stage = int(self.steps / self.stages)
        force_inc = self.maxForce / steps_per_stage

        # Load
        for step in range(1, steps_per_stage+1):

            print('--------------------------------------')
            print('LOAD STEP ', globalStep)
            force -= force_inc
            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)
            if solverSuccess == False:
                raise ValueError('Solver failed to converge.')

            store_force_displacement(Uu, force, fd_force, fd_disp)

            self.state.append((Uu, p))

            if self.writeOutput:
              write_vtk_output(Uu, p, globalStep + 1)

            globalStep += 1

        # Unload
        for step in range(1, steps_per_stage+1):
            print('--------------------------------------')
            print('LOAD STEP ', globalStep)
            force += force_inc
            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)
            if solverSuccess == False:
                raise ValueError('Solver failed to converge.')

            store_force_displacement(Uu, force, fd_force, fd_disp)

            self.state.append((Uu, p))

            if self.writeOutput:
              write_vtk_output(Uu, p, globalStep + 1)

            globalStep += 1

        self.stateNotStored = False

    def setup_energy_functions(self):
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quad_rule.xigauss)

        def energy_function_all_dofs(U, p, props):
            internal_variables = p.state_data
            #adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(props, shapeOnRef, self.mesh, self.quad_rule)
            #mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.mat_model)
            return self.mech_funcs.compute_strain_energy(U, internal_variables, props)

        nodal_forces = jit(grad(energy_function_all_dofs, argnums=0))

        def energy_function_props(Uu, p, props):
            U = self.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, p, props)

        return EnergyFunctions(energy_function_props, nodal_forces)


    def compute_L2_norm_difference(self, uSteps, pSteps, properties, nodal_forces):
        numerator = 0.0
        denominator= 0.0
        for i in range(0, len(self.targetSteps)):
            step = self.targetSteps[i]
            Uu = uSteps[step]
            p = pSteps[step]

            U = self.create_field(Uu, p.bc_data)
            force = np.sum(np.array(nodal_forces(U, p, properties).at[self.index].get()))
            diff = force - self.targetForces[i]
            numerator += diff*diff
            denominator += self.targetForces[i]*self.targetForces[i]

        return np.sqrt(numerator/denominator)

    def get_objective(self):
        if self.stateNotStored:
            self.run_simulation()

        parameters = self.elementProperties
        energyFuncs = self.setup_energy_functions()
        
        uSteps = np.stack([self.state[i][0] for i in range(0, self.steps+1)], axis=0)
        pSteps = [self.state[i][1] for i in range(0, self.steps+1)]
        
        val = self.compute_L2_norm_difference(uSteps, pSteps, parameters, energyFuncs.nodal_forces) 
     
        return onp.array(self.scaleObjective * val).item()      

    def get_gradient(self):
        if self.stateNotStored:
            self.run_simulation()

        energyFuncs = self.setup_energy_functions()
        residualInverseFuncs = MechanicsInverse.create_residual_inverse_functions(energyFuncs.energy_function_props)
 
        # derivatives of F
        parameters = self.elementProperties
        uSteps = np.stack([self.state[i][0] for i in range(0, self.steps+1)], axis=0)
        pSteps = [self.state[i][1] for i in range(0, self.steps+1)]
        df_du, gradient = grad(self.compute_L2_norm_difference, (0, 2))(uSteps, pSteps, parameters, energyFuncs.nodal_forces)
 
        adjointLoad = np.zeros(uSteps[0].shape)
 
        for step in reversed(range(1, self.steps+1)):
            Uu = uSteps[step]
            p = pSteps[step]
 
            adjointLoad -= df_du[step]
 
            n = self.dof_manager.get_unknown_size()
            p_objective = Objective.Params(bc_data=p.bc_data, state_data=p.state_data) # remember R is a function of ivs_prev
            self.objective.p = p_objective 
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), rtol=1e-10, atol=0.0, M=dRdu_decomp)[0]
 
            gradient += residualInverseFuncs.residual_jac_coords_vjp(Uu, p, parameters, adjointVector)
 
            adjointLoad = np.zeros(uSteps[0].shape)
 
        return onp.array(self.scaleObjective * gradient[self.elementMap], copy=False).flatten().tolist()

        # print(type(gradient))
        # print(gradient.shape)

        # return onp.array(self.scaleObjective * gradient, self.dprops_ddensity, copy=False).flatten()[self.elementMap].tolist()

if __name__ == '__main__':
    mpo = MaterialPropertiesOptimization()

    # Import dummy parameters
    densityValue = 0.4
    materialProperties = np.full((mpo.mesh.conns.shape[0]), densityValue).tolist()
    mpo.import_parameters(materialProperties)

    val = mpo.get_objective()
    print("\n objective value")
    print(val)
    # print("\n gradient")
    # grad = mpo.get_gradient()
    # print(grad)
