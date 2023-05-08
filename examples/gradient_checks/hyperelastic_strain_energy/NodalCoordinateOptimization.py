from jax import grad
from jax import jit
from jax import value_and_grad
from optimism import EquationSolver
from optimism import ExodusWriter
from optimism import FunctionSpace
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism.FunctionSpace import DofManager
from optimism.FunctionSpace import EssentialBC
from optimism.material import Neohookean
from typing import Optional

import matplotlib.pyplot as plt

import jax.numpy as np
import numpy as onp

class NodalCoordinateOptimization:

    def __init__(self):
        self.stateNotStored = True

        self.quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)

        self.ebcs = [
            EssentialBC(nodeSet='yminus_nodeset', component=0),
            EssentialBC(nodeSet='yminus_nodeset', component=1),
            EssentialBC(nodeSet='yplus_nodeset', component=0),
            EssentialBC(nodeSet='yplus_nodeset', component=1)
        ]

        props = {
            'elastic modulus': 3. * 10.0 * (1. - 2. * 0.3),
            'poisson ratio': 0.3,
            'version': 'coupled'
        }
        self.mat_model = Neohookean.create_material_model_functions(props)

        self.eq_settings = EquationSolver.get_settings(
            use_incremental_objective=False,
            max_trust_iters=100,
            tr_size=0.25,
            min_tr_size=1e-15,
            tol=5e-8
        )

        self.input_mesh = './window.exo'
        # self.output_file = 'output.exo'
        self.plot_file = 'disp_control_response.npz'
        self.steps = 20
        self.maxDisp = 0.25

    def reload_mesh(self):
        self.mesh = ReadExodusMesh.read_exodus_mesh(self.input_mesh)
        self.stateNotStored = True

    def run_simulation(self):

        coords = self.mesh.coords

        # setup
        func_space = FunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quad_rule)
        mech_funcs = Mechanics.create_mechanics_functions(func_space, mode2D='plane strain', materialModel=self.mat_model)
        dof_manager = DofManager(func_space, 2, self.ebcs)

        # methods defined on the fly
        def get_ubcs(p):
            disp = p[0]
            V = np.zeros(coords.shape)
            index = (self.mesh.nodeSets['yplus_nodeset'], 1)
            V = V.at[index].set(disp)
            return dof_manager.get_bc_values(V)

        def create_field(Uu, p):
            return dof_manager.create_field(Uu, get_ubcs(p))

        def energy_function(Uu, p):
            U = create_field(Uu, p)
            internal_variables = p[1]
            return mech_funcs.compute_strain_energy(U, internal_variables)
        
        def energy_function_alt(U, p):
            internal_variables = p[1]
            return mech_funcs.compute_strain_energy(U, internal_variables)

        nodal_forces = jit(grad(energy_function_alt, argnums=0))

        def assemble_sparse(Uu, p):
            U = create_field(Uu, p)
            internal_variables = p[1]
            element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables)
            return SparseMatrixAssembler.\
                assemble_sparse_stiffness_matrix(element_stiffnesses, func_space.mesh.conns, dof_manager)
    
        def store_force_displacement(Uu, dispval, force, disp):
            U = create_field(Uu, p)
            f = nodal_forces(U, p)

            index = (self.mesh.nodeSets['yplus_nodeset'], 1)
            force.append( onp.abs(onp.sum(onp.array(f.at[index].get()))) )

            disp.append( onp.abs(dispval) )

            with open(self.plot_file,'wb') as f:
                np.savez(f, force=force, displacement=disp)

        # only call after calculations are finished
        def save_displacement(Uu, exo, step):
            exo.put_time(step, step)
            U = create_field(Uu, p)
            ExodusWriter.write_exodus_nodal_outputs(
                exo,
                ['disp_x', 'disp_y'], [U[:, 0], U[:, 1]], time_step=step)

        # problem set up
        Uu = dof_manager.get_unknown_values(np.zeros(coords.shape))
        ivs = mech_funcs.compute_initial_state()
        p = Objective.Params(0., ivs)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        # # set up output mesh
        # ExodusWriter.copy_exodus_mesh(self.input_mesh, self.output_file)
        # exo = ExodusWriter.setup_exodus_database(
        #     self.output_file,
        #     2, 0, ['disp_x', 'disp_y'], []
        # )
        # save_displacement(Uu, exo, 0)

        # loop over load steps
        disp = 0.
        fd_force = []
        fd_disp = []

        store_force_displacement(Uu, disp, fd_force, fd_disp)
        for step in range(1, self.steps+1):

            print('--------------------------------------')
            print('LOAD STEP ', step)
            disp = disp - self.maxDisp / self.steps
            p = Objective.param_index_update(p, 0, disp)
            Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, self.eq_settings)

            store_force_displacement(Uu, disp, fd_force, fd_disp)
            # save_displacement(Uu, exo, step)

        self.state = (Uu, p)
        self.stateNotStored = False

    def objective_function(self, coords):
        f_space = FunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quad_rule)
        m_funcs = Mechanics.create_mechanics_functions(f_space, mode2D='plane strain', materialModel=self.mat_model)

        dof_manager = DofManager(f_space, 2, self.ebcs)
        def get_ubcs(p):
            disp = p[0]
            V = np.zeros(coords.shape)
            index = (self.mesh.nodeSets['yplus_nodeset'], 1)
            V = V.at[index].set(disp)
            return dof_manager.get_bc_values(V)

        U = dof_manager.create_field(self.state[0], get_ubcs(self.state[1]))
        state = self.state[1][1]

        return m_funcs.compute_strain_energy(U, state)
    
    def get_objective(self):
        if self.stateNotStored:
            self.run_simulation()

        value = -self.objective_function(self.mesh.coords) 
        return onp.array(value).item()        


    def get_gradient(self):
        if self.stateNotStored:
            self.run_simulation()

        gradient = -grad(self.objective_function, argnums=0)(self.mesh.coords)
        return onp.array(gradient).flatten().tolist()
