from jax import grad
from jax import jit
from optimism import EquationSolver
from plato_optimism import exodus_writer as ExodusWriter
from plato_optimism import GradUtilities
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
from typing import Optional
from collections import namedtuple

import jax.numpy as np
import numpy as onp
from scipy.sparse import linalg
from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace

EnergyFunctions = namedtuple('EnergyFunctions',
                            ['energy_function_coords',
                             'nodal_forces'])

class NodalCoordinateOptimization:

    def __init__(self):
        self.writeOutput = True

        self.scaleObjective = -1.0 # -1.0 to maximize
        self.stateNotStored = True

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
        props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        self.mat_model = HyperViscoelastic.create_material_model_functions(props)

        self.eq_settings = EquationSolver.get_settings(
            use_incremental_objective=False,
            max_trust_iters=100,
            tr_size=0.25,
            min_tr_size=1e-15,
            tol=5e-8
        )

        self.input_mesh = './hole_array.exo'
        if self.writeOutput:
          self.output_file = 'output.exo'

        weissenbergNumber = 1.0e-2
        strain = 0.1 # height is 50
        stageTime = strain * tau_1 / weissenbergNumber

        self.maxDisp = 5.0
        self.plot_file = 'disp_control_response.npz'
        self.stages = 2
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
            internal_variables = p[1]
            return mech_funcs.compute_strain_energy(U, internal_variables, self.dt)

        def energy_function(Uu, p):
            U = self.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, p)

        nodal_forces = jit(grad(energy_function_all_dofs, argnums=0))

        def assemble_sparse(Uu, p):
            U = self.create_field(Uu, p.bc_data)
            internal_variables = p[1]
            element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables, self.dt)
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

        def save_exodus_outputs(Uu, p, exo, step):
            exo.put_time(step, step)
            U = self.create_field(Uu, p.bc_data)
            ExodusWriter.write_exodus_nodal_outputs(
                exo,
                node_variable_names=['disp_x', 'disp_y'], 
                node_variable_values=[U[:, 0], U[:, 1]], 
                time_step=step)

            energyDensities = mech_funcs.compute_output_energy_densities_and_stresses(U, p.state_data, self.dt)[0]
            dissipationEnergyDensities = mech_funcs.compute_output_material_qoi(U, p.state_data, self.dt)
            cellDissipationEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, dissipationEnergyDensities)
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, energyDensities)
            ExodusWriter.write_exodus_element_outputs(
                exo,
                element_variable_names=['strain energy density', 'dissipation density'],
                element_variable_values=[cellEnergyDensities, cellDissipationEnergyDensities], 
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
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)

            U = self.create_field(Uu, p.bc_data)
            ivs = mech_funcs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            store_force_displacement(Uu, disp, fd_force, fd_disp)
            self.state.append((Uu, p))

            if self.writeOutput:
              save_exodus_outputs(Uu, p, exo, globalStep + 1)
            
            globalStep += 1

        # Unload
        for step in range(1, steps_per_stage+1):
            print('--------------------------------------')
            print('LOAD STEP ', globalStep)
            disp += disp_inc
            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)

            U = self.create_field(Uu, p.bc_data)
            ivs = mech_funcs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            store_force_displacement(Uu, disp, fd_force, fd_disp)
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
            U = self.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, ivs, coords)

        nodal_forces = grad(energy_function_all_dofs, argnums=0)

        return EnergyFunctions(energy_function_coords, jit(nodal_forces))

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

    def get_gradient(self):
        if self.stateNotStored:
            self.run_simulation()

        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        energyFuncs = self.setup_energy_functions()
        ivsUpdateInverseFuncs = MechanicsInverse.create_ivs_update_inverse_functions(functionSpace,
                                                                                     "plane strain",
                                                                                     self.mat_model)
        residualInverseFuncs = MechanicsInverse.create_path_dependent_residual_inverse_functions(energyFuncs.energy_function_coords)

        # derivatives of F
        parameters = self.mesh.coords
        uSteps = np.stack([self.state[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([self.state[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [self.state[i][1] for i in range(0, self.steps+1)]
        df_du, df_dc, gradient = grad(self.compute_total_work, (0, 2, 3))(uSteps, pSteps, ivsSteps, parameters, energyFuncs.nodal_forces) 

        mu = np.zeros(ivsSteps[0].shape) 
        adjointLoad = np.zeros(uSteps[0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = uSteps[step]
            p = pSteps[step]
            U = self.create_field(Uu, p.bc_data)
            ivs_prev = ivsSteps[step-1]

            dc_dcn = ivsUpdateInverseFuncs.ivs_update_jac_ivs_prev(U, ivs_prev, self.dt)

            mu += df_dc[step]
            adjointLoad -= df_du[step]
            adjointLoad -= self.dof_manager.get_unknown_values(ivsUpdateInverseFuncs.ivs_update_jac_disp_vjp(U, ivs_prev, mu, self.dt))

            n = self.dof_manager.get_unknown_size()
            p_objective = Objective.Params(bc_data=p.bc_data, state_data=ivs_prev) # remember R is a function of ivs_prev
            self.objective.p = p_objective 
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), rtol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            gradient += residualInverseFuncs.residual_jac_coords_vjp(Uu, p, ivs_prev, parameters, adjointVector)
            gradient += ivsUpdateInverseFuncs.ivs_update_jac_coords_vjp(U, ivs_prev, parameters, mu, self.dt)

            mu = np.einsum('ijk,ijkn->ijn', mu, dc_dcn)
            mu += residualInverseFuncs.residual_jac_ivs_prev_vjp(Uu, p, ivs_prev, parameters, adjointVector)

            adjointLoad = np.zeros(uSteps[0].shape)

        projectedGrad = GradUtilities.projectToVertices(self.mesh, gradient)
        return onp.array(self.scaleObjective * projectedGrad, copy=False).flatten().tolist()



if __name__ == '__main__':
    nco = NodalCoordinateOptimization()
    nco.reload_mesh()

    val = nco.get_objective()
    print("\n Objective is: ")
    print(val)

    # grad = nco.get_gradient()
    # print(f"\n Gradient is: {grad}")
    # with open('gradient_out.npz','wb') as f:
    #     np.savez(f, grad=grad) 
