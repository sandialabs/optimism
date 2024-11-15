from collections import namedtuple
from jax import grad
from jax import jit
from optimism import EquationSolver
# from plato_optimism import exodus_writer as ExodusWriter
# from plato_optimism import GradUtilities
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism import VTKWriter
from optimism.FunctionSpace import DofManager
from optimism.FunctionSpace import EssentialBC
from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace
from optimism.material import Neohookean_VariableProps
from optimism.material import HyperViscoelastic_VariableProps
import jax.numpy as np
import numpy as onp
from scipy.sparse import linalg
from typing import Callable, NamedTuple


class EnergyFunctions(NamedTuple):
    energy_function_coords: Callable
    nodal_forces: Callable


class ElementPropertyOptimization:
    def __init__(self):
        self.writeOutput = True
        self.scaleObjective = -1.0 # -1.0 to maximize
        self.stateNotStored = True

        self.input_mesh = './ellipse_test_LD_sinusoidal_1.exo'
        if self.writeOutput:
          self.output_file = 'output_ellipse_VARIABLE_2.exo'

        self.quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)

        self.ebcs = [
            EssentialBC(nodeSet='yminus_sideset', component=0),
            EssentialBC(nodeSet='yminus_sideset', component=1),
            EssentialBC(nodeSet='yplus_sideset', component=0),
            EssentialBC(nodeSet='yplus_sideset', component=1)
        ]
        
        G_eq = 0.855 # MPa
        K_eq = 1000*G_eq # MPa
        G_neq_1 = 4.0*G_eq
        tau_1   = 0.1
        constant_props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        varProps = ReadExodusMesh.read_exodus_mesh_element_properties(self.input_mesh, ['light_dose'], blockNum=1)
        # print(varProps.shape)
        # assert False

        # props = {
        #     'youngs modulus': 1.0,
        #     'poisson ratio': 0.3
        # }
        
        self.mat_model = HyperViscoelastic_VariableProps.create_material_model_functions(constant_props)
        # constant_props = {
        #     'density': 1.0
        # }
        # self.mat_model = Neohookean_VariableProps.create_material_model_functions(constant_props, 'adagio')

        self.eq_settings = EquationSolver.get_settings(
            use_incremental_objective=False,
            max_trust_iters=100,
            tr_size=0.25,
            min_tr_size=1e-15,
            tol=5e-8
        )
        stageTime = 0.05
        self.maxDisp = 0.2
        self.plot_file = 'disp_control_response.npz'
        self.stages = 1
        steps_per_stage = 40
        self.steps = self.stages * steps_per_stage
        self.dt = stageTime / steps_per_stage

    def create_field(self, Uu, disp):
        def get_ubcs(disp):
            V = np.zeros(self.mesh.coords.shape)
            index = (self.mesh.nodeSets['yplus_sideset'], 1)
            V = V.at[index].set(disp)
            return self.dof_manager.get_bc_values(V)

        return self.dof_manager.create_field(Uu, get_ubcs(disp))

    def reload_mesh(self):
        origMesh = ReadExodusMesh.read_exodus_mesh(self.input_mesh)
        # TODO
        # really this will be handed off from plato
        # using exodus as a surrogate for now.
        # Might need some reshape operation here 
        self.props = ReadExodusMesh.read_exodus_mesh_element_properties(self.input_mesh, ['light_dose'], blockNum=1)
        # TODO
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(origMesh, order=2, createNodeSetsFromSideSets=True)

        func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        self.dof_manager = DofManager(func_space, 2, self.ebcs)

        self.stateNotStored = True
        self.state = []

    def run_simulation(self):

        # methods defined on the fly
        func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        mech_funcs = Mechanics.create_mechanics_functions(func_space, mode2D='plane strain', materialModel=self.mat_model)

        def energy_function_all_dofs(U, p):
            internal_variables = p.state_data
            props = p.prop_data
            return mech_funcs.compute_strain_energy(U, internal_variables, props, self.dt)

        def energy_function(Uu, p):
            U = self.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, p)

        nodal_forces = jit(grad(energy_function_all_dofs, argnums=0))

        def assemble_sparse(Uu, p):
            U = self.create_field(Uu, p.bc_data)
            internal_variables = p.state_data
            props = p.prop_data
            element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables, props, self.dt)
            return SparseMatrixAssembler.\
                assemble_sparse_stiffness_matrix(element_stiffnesses, func_space.mesh.conns, self.dof_manager)
    
        def store_force_displacement(Uu, dispval, force, disp):
            U = self.create_field(Uu, p.bc_data)
            f = nodal_forces(U, p)

            index = (self.mesh.nodeSets['yplus_sideset'], 1)

            force.append( onp.sum(onp.array(f.at[index].get())) )
            disp.append( dispval )

            with open(self.plot_file,'wb') as f:
                np.savez(f, force=force, displacement=disp)

        # def save_exodus_outputs(Uu, p, exo, step):
        def save_exodus_outputs(Uu, p, step):
            # exo.put_time(step, step)
            U = self.create_field(Uu, p.bc_data)
            # ExodusWriter.write_exodus_nodal_outputs(
            #     exo,
            #     node_variable_names=['disp_x', 'disp_y'], 
            #     node_variable_values=[U[:, 0], U[:, 1]], 
            #     time_step=step
            # )

            # energyDensities = mech_funcs.compute_output_energy_densities_and_stresses(U, p.state_data, p.prop_data, self.dt)[0]
            # # dissipationEnergyDensities = mech_funcs.compute_output_material_qoi(U, p.state_data, self.dt)
            # # cellDissipationEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, dissipationEnergyDensities)
            # cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, energyDensities)
            # ExodusWriter.write_exodus_element_outputs(
            #     exo,
            #     # element_variable_names=['strain energy density', 'dissipation density'],
            #     element_variable_names=['strain energy density'],
            #     # element_variable_values=[cellEnergyDensities, cellDissipationEnergyDensities], 
            #     element_variable_values=[cellEnergyDensities],
            #     time_step=step, block_id=1
            # )
            plotName = 'output-'+str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

            # bcs = np.array(self.dofManager.isBc, dtype=int)
            # writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

            # Ubc = self.get_ubcs(p)
            # internalVariables = p[1]
            # rxnBc = self.compute_bc_reactions(Uu, Ubc, p)
            # reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
            # writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

            # energyDensities, stresses = mech_funcs.compute_output_energy_densities_and_stresses(U, internalVariables)
            # cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, energyDensities)
            # cellStresses = FunctionSpace.project_quadrature_field_to_element_field(self.fs, stresses)
            # writer.add_cell_field(name='strain_energy_density',
            #                     cellData=cellEnergyDensities,
            #                     fieldType=VTKWriter.VTKFieldType.SCALARS)
            # writer.add_cell_field(name='piola_stress',
            #                     cellData=cellStresses,
            #                     fieldType=VTKWriter.VTKFieldType.TENSORS)

            writer.write()

        # problem set up
        Uu = self.dof_manager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mech_funcs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs, prop_data=self.props)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        self.objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        # set up output mesh
        if self.writeOutput:
            # ExodusWriter.copy_exodus_mesh(self.input_mesh, self.output_file)
            # exo = ExodusWriter.setup_exodus_database(
            #     self.output_file,
            #     num_node_variables=2, 
            #     num_element_variables=1, 
            #     node_variable_names=['disp_x', 'disp_y'], 
            #     # element_variable_names=['strain energy density', 'dissipation density']
            #     element_variable_names=['strain energy density']
            # )
            # save_exodus_outputs(Uu, p, exo, 1)
            save_exodus_outputs(Uu, p, 1)

        # loop over load steps
        disp = 0.
        fd_force = []
        fd_disp = []

        store_force_displacement(Uu, disp, fd_force, fd_disp)
        self.state.append((Uu, p))

        globalStep = 1

        # Load
        steps_per_stage = int(self.steps / self.stages)
        disp_inc = self.maxDisp / steps_per_stage
        for step in range(1, steps_per_stage+1):
            print('--------------------------------------')
            print('LOAD STEP ', globalStep)
            disp -= disp_inc
            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings, useWarmStart=False)

            U = self.create_field(Uu, p.bc_data)
            ivs = mech_funcs.compute_updated_internal_variables(U, p.state_data, p.prop_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)
            store_force_displacement(Uu, disp, fd_force, fd_disp)
            self.state.append((Uu, p))

            if self.writeOutput:
                save_exodus_outputs(Uu, p, globalStep + 1)
            #   save_exodus_outputs(Uu, p, exo, globalStep + 1)
            
            globalStep += 1

        self.stateNotStored = False

    def setup_energy_functions(self):
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quad_rule.xigauss)

        # TODO add props as a input to compute_strain_energy method
        def energy_function_all_dofs(U, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quad_rule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.mat_model)
            # TODO add props as a input to compute_strain_energy method
            return mech_funcs.compute_strain_energy(U, ivs, self.props, self.dt)

        # TODO add props as a input to compute_strain_energy method? Maybe?
        def energy_function_coords(Uu, p, ivs, coords):
            U = self.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, ivs, coords)

        nodal_forces = grad(energy_function_all_dofs, argnums=0)

        return EnergyFunctions(energy_function_coords, jit(nodal_forces))

    # Let Ryan figure out what he needs here...
    def compute_total_work(self, uSteps, pSteps, ivsSteps, coordinates, nodal_forces):
        index = (self.mesh.nodeSets['yplus_sideset'], 1)

        totalWork = 0.0
        for step in range(1, self.steps+1):
            Uu = uSteps[step]
            p = pSteps[step]
            ivs = ivsSteps[step]
            U = self.create_field(Uu, p.bc_data)
            force = np.array(nodal_forces(U, ivs, coordinates).at[index].get())
            disp = U.at[index].get()

            Uu_prev = uSteps[step-1]
            p_prev = pSteps[step-1]
            ivs_prev = ivsSteps[step-1]
            U_prev = self.create_field(Uu_prev, p_prev.bc_data)
            force_prev = np.array(nodal_forces(U_prev, ivs_prev, coordinates).at[index].get())
            disp_prev = U_prev.at[index].get()

            totalWork += 0.5*np.tensordot((force + force_prev),(disp - disp_prev), axes=1)

        return totalWork

    def get_objective(self):
        if self.stateNotStored:
            self.run_simulation()

        parameters = self.mesh.coords
        energyFuncs = self.setup_energy_functions()
        uSteps = np.stack([self.state[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([self.state[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [self.state[i][1] for i in range(0, self.steps+1)]

        val = self.compute_total_work(uSteps, pSteps, ivsSteps, parameters, energyFuncs.nodal_forces) 
        return onp.array(self.scaleObjective * val).item()        


if __name__ == '__main__':
    nco = ElementPropertyOptimization()
    nco.reload_mesh()
    val = nco.get_objective()
    print("\n Objective is: ")
    print(val)
