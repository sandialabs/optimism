from jax import grad
from jax import jit
from optimism import EquationSolver
from plato_optimism import exodus_writer as ExodusWriter
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
from optimism.material import HyperViscoelastic
from collections import namedtuple

import jax.numpy as np
import numpy as onp
from optimism.inverse import AdjointFunctionSpace

EnergyFunctions = namedtuple('EnergyFunctions',
                            ['energy_function_coords',
                             'nodal_forces',
                             'dissipation'])

class NodalCoordinateOptimization:

    def __init__(self):
        self.writeOutput = True

        self.quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        self.ebcs = [
            EssentialBC(nodeSet='bottom_sideset', component=1),

            EssentialBC(nodeSet='right_sideset', component=0),
            EssentialBC(nodeSet='left_sideset', component=0),

            # EssentialBC(nodeSet='left_roller_sideset', component=0),
            # EssentialBC(nodeSet='right_roller_sideset', component=0),
        ]

        G_eq = 0.855 # MPa
        K_eq = 1000*G_eq # MPa
        G_neq_1 = 4.0*G_eq
        tau_1   = 0.1
        props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        self.mat_model = HyperViscoelastic.create_material_model_functions(props)

        self.eq_settings = EquationSolver.get_settings(max_trust_iters=1100, use_preconditioned_inner_product_for_cg=True)

        self.input_mesh = './snap_cell.exo'
        if self.writeOutput:
          self.output_file = 'output.exo'

        weissenbergNumber = 1.0e-4
        base_thickness = 35.0 
        pillar_height = 54.0 
        apex_height = 24.0 
        totalHeight = base_thickness + pillar_height + apex_height + pillar_height
        maxDisp = 65
        strain = maxDisp / totalHeight
        stageTime = strain * tau_1 / weissenbergNumber

        self.maxForce = -1.5 # 1.5, 1.5, 2.0, 2.5 (Wi=1E-6, 1E-4, 1E-3, 1E-2)
        self.plot_file = 'force_control_response.npz'
        self.stages = 2
        steps_per_stage = 65
        self.steps = self.stages * steps_per_stage
        self.dt = stageTime / steps_per_stage

    def create_field(self, Uu):
        def get_ubcs():
            V = np.zeros(self.mesh.coords.shape)
            return self.dof_manager.get_bc_values(V)

        return self.dof_manager.create_field(Uu, get_ubcs())

    def reload_mesh(self):
        origMesh = ReadExodusMesh.read_exodus_mesh(self.input_mesh)
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(origMesh, order=2, createNodeSetsFromSideSets=True)

        func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        self.dof_manager = DofManager(func_space, 2, self.ebcs)

        surfaceXCoords = self.mesh.coords[self.mesh.nodeSets['top_sideset']][:,0]
        self.tractionArea = np.max(surfaceXCoords) - np.min(surfaceXCoords)

        self.stateNotStored = True
        self.state = []

    def run_simulation(self):

        # methods defined on the fly
        func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        mech_funcs = Mechanics.create_mechanics_functions(func_space, mode2D='plane strain', materialModel=self.mat_model)

        def energy_function(Uu, p):
            U = self.create_field(Uu)
            internal_variables = p.state_data
            strainEnergy = mech_funcs.compute_strain_energy(U, internal_variables, self.dt)

            F = p.bc_data
            def force_function(x, n):
                return np.array([0.0, F/self.tractionArea])

            loadPotential = Mechanics.compute_traction_potential_energy(
                func_space, U, self.lineQuadRule, self.mesh.sideSets['top_sideset'], 
                force_function)

            return strainEnergy + loadPotential

        def assemble_sparse(Uu, p):
            U = self.create_field(Uu)
            internal_variables = p.state_data
            element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables, self.dt)
            return SparseMatrixAssembler.\
                assemble_sparse_stiffness_matrix(element_stiffnesses, func_space.mesh.conns, self.dof_manager)
    
        def store_force_displacement(Uu, force_val, force, disp):
            U = self.create_field(Uu)

            index = (self.mesh.nodeSets['top_sideset'], 1)

            force.append( force_val )
            disp.append(np.mean(U.at[index].get()))

            with open(self.plot_file,'wb') as f:
                np.savez(f, force=force, displacement=disp)

        def save_exodus_outputs(Uu, p, exo, step):
            exo.put_time(step, step)
            U = self.create_field(Uu)
            ExodusWriter.write_exodus_nodal_outputs(
                exo,
                node_variable_names=['disp_x', 'disp_y'], 
                node_variable_values=[U[:, 0], U[:, 1]], 
                time_step=step)

            energyDensities = mech_funcs.compute_output_energy_densities_and_stresses(U, p.state_data, self.dt)[0]
            dissipationDensities = mech_funcs.compute_output_material_qoi(U, p.state_data, self.dt)
            cellDissipationDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, dissipationDensities)
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, energyDensities)
            ExodusWriter.write_exodus_element_outputs(
                exo,
                element_variable_names=['strain energy density', 'dissipation density'],
                element_variable_values=[cellEnergyDensities, cellDissipationDensities], 
                time_step=step, block_id=1)

        # problem set up
        Uu = self.dof_manager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mech_funcs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        self.objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        # set up output mesh
        if self.writeOutput:
            ExodusWriter.copy_exodus_mesh(self.input_mesh, self.output_file)
            exo = ExodusWriter.setup_exodus_database(
                self.output_file,
                num_node_variables=2, num_element_variables=2, 
                node_variable_names=['disp_x', 'disp_y'], 
                element_variable_names=['strain energy density', 'dissipation density']
            )
            save_exodus_outputs(Uu, p, exo, 1)

        # loop over load steps
        force = 0.
        fd_force = []
        fd_disp = []

        store_force_displacement(Uu, force, fd_force, fd_disp)
        self.state.append((Uu, p))

        globalStep = 1

        # Load
        steps_per_stage = int(self.steps / self.stages)
        force_inc = self.maxForce / steps_per_stage
        for step in range(1, steps_per_stage+1):
            print('--------------------------------------')
            print('LOAD STEP ', globalStep)
            force += force_inc
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)

            U = self.create_field(Uu)
            ivs = mech_funcs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            store_force_displacement(Uu, force, fd_force, fd_disp)
            self.state.append((Uu, p))

            if self.writeOutput:
              save_exodus_outputs(Uu, p, exo, globalStep + 1)

            globalStep += 1

        # Unload
        for step in range(1, steps_per_stage+1):
            print('--------------------------------------')
            print('LOAD STEP ', globalStep)
            force -= force_inc
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)

            U = self.create_field(Uu)
            ivs = mech_funcs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            store_force_displacement(Uu, force, fd_force, fd_disp)
            self.state.append((Uu, p))

            if self.writeOutput:
              save_exodus_outputs(Uu, p, exo, globalStep + 1)

            globalStep += 1

        self.stateNotStored = False
    def setup_energy_functions(self):
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quad_rule.xigauss)

        def energy_function_all_dofs(U, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quad_rule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.mat_model)
            return mech_funcs.compute_strain_energy(U, ivs, self.dt)

        def energy_function_coords(Uu, p, ivs, coords):
            U = self.create_field(Uu)
            return energy_function_all_dofs(U, ivs, coords)

        nodal_forces = grad(energy_function_all_dofs, argnums=0)

        def dissipation(Uu, p, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quad_rule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.mat_model)

            U = self.create_field(Uu)
            return mech_funcs.integrated_material_qoi(U, ivs, self.dt)

        return EnergyFunctions(energy_function_coords, jit(nodal_forces), dissipation)

    def compute_energy_quantities(self, uSteps, pSteps, ivsSteps, coordinates, nodal_forces, energy_function_coords, compute_dissipation):
        index = (self.mesh.nodeSets['top_sideset'], 1)

        totalWork = 0.0
        dissipation = 0.0

        totalWorkStored = []
        strainEnergyStored = []
        dissipationStored = []
        dissipation2Stored = []
        timeStored = []

        for step in range(1, self.steps+1):
            Uu = uSteps[step]
            p = pSteps[step]
            ivs = ivsSteps[step]
            U = self.create_field(Uu)
            force = p.bc_data
            disp = np.mean(U.at[index].get())

            Uu_prev = uSteps[step-1]
            p_prev = pSteps[step-1]
            U_prev = self.create_field(Uu_prev)
            force_prev = p_prev.bc_data
            disp_prev = np.mean(U_prev.at[index].get())

            totalWork += 0.5*(force + force_prev)*(disp - disp_prev)

            dissipation = totalWork - energy_function_coords(Uu, p, ivs, coordinates)
            dissipation2 = self.dt * compute_dissipation(Uu, p, ivs, coordinates) # right Reimann sum (rectangle rule)

            totalWorkStored.append(totalWork)
            dissipationStored.append(dissipation)
            dissipation2Stored.append(dissipation2)
            strainEnergyStored.append(energy_function_coords(Uu, p, ivs, coordinates))
            timeStored.append(self.dt)

        with open("energy_histories.npz",'wb') as f:
            np.savez(f, totalWork=totalWorkStored, dissipation=dissipationStored, dissipation2=dissipation2Stored, strainEnergy=strainEnergyStored, time=timeStored)

    def compute_output(self):
        if self.stateNotStored:
            self.run_simulation()

        parameters = self.mesh.coords
        energyFuncs = self.setup_energy_functions()

        uSteps = np.stack([self.state[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([self.state[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [self.state[i][1] for i in range(0, self.steps+1)]

        self.compute_energy_quantities(uSteps, pSteps, ivsSteps, parameters, energyFuncs.nodal_forces, jit(energyFuncs.energy_function_coords), jit(energyFuncs.dissipation)) 



if __name__ == '__main__':
    nco = NodalCoordinateOptimization()
    nco.reload_mesh()
    nco.compute_output()

