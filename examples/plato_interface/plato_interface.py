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

import jax.numpy as np
import numpy as onp

# constants up here
#
quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
ebcs = [
    EssentialBC(nodeSet='nset_outer_bottom', component=0),
    EssentialBC(nodeSet='nset_outer_bottom', component=1),
    EssentialBC(nodeSet='nset_outer_top', component=0),
    EssentialBC(nodeSet='nset_outer_top', component=1)
]
props = {
    'elastic modulus': 3. * 10.0 * (1. - 2. * 0.3),
    'poisson ratio': 0.3,
    'version': 'coupled'
}
mat_model = Neohookean.create_material_model_functions(props)
eq_settings = EquationSolver.get_settings(
    use_incremental_objective=False,
    max_trust_iters=100,
    tr_size=0.25,
    min_tr_size=1e-15,
    tol=5e-8
)
input_mesh = './geometry.g'
output_file = './output.e'
steps = 10
maxDisp = 0.25


def run_simulation(coords: np.ndarray, mesh: Mesh):

    # setup
    func_space = FunctionSpace.construct_function_space_for_adjoint(coords, mesh, quad_rule)
    mech_funcs = Mechanics.create_mechanics_functions(func_space, mode2D='plane strain', materialModel=mat_model)
    dof_manager = DofManager(func_space, 2, ebcs)

    # methods defined on the fly
    def get_ubcs(p):
        disp = p[0]
        V = np.zeros(coords.shape)
        index = (mesh.nodeSets['nset_outer_top'], 1)
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


    # only call after calculations are finished
    def objective_function(coords, Uu, p):
        f_space = FunctionSpace.construct_function_space_for_adjoint(coords, mesh, quad_rule)
        m_funcs = Mechanics.create_mechanics_functions(f_space, mode2D='plane strain', materialModel=mat_model)
        U = create_field(Uu, p)
        state = p[1]
        return m_funcs.compute_strain_energy(U, state)

    # problem set up
    Uu = dof_manager.get_unknown_values(np.zeros(coords.shape))
    ivs = mech_funcs.compute_initial_state()
    p = Objective.Params(0., ivs)
    precond_strategy = Objective.PrecondStrategy(assemble_sparse)
    objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

    # loop over load steps
    disp = 0.
    for step in range(1, steps):
        print('--------------------------------------')
        print('LOAD STEP ', step)
        disp = disp - maxDisp / steps
        p = Objective.param_index_update(p, 0, disp)
        Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, eq_settings)

        """
        # post-processing for debugging or other stuff
        #
        # Ryan, this should be most outputs you would want to play with
        # so use these as a starting point to play with other objectives
        # U - displacements
        # f - nodal forces (including reaction forces at essential bcs)
        # psis - element averaged (I think) strain energy density
        # stresses - element averaged (I think) cauchy stresses
        #
        # U = create_field(Uu, p)
        # f = nodal_forces(U, p)
        # state = mech_funcs.compute_updated_internal_variables(U, p.state_data)
        # psis, stresses = mech_funcs.compute_output_energy_densities_and_stresses(U, state)
        # psis = FunctionSpace.project_quadrature_field_to_element_field(func_space, psis)
        # stresses = FunctionSpace.project_quadrature_field_to_element_field(func_space, stresses)
        """

    return objective_function(coords, Uu, p), grad(objective_function, argnums=0)(coords, Uu, p)



gradients = grad(run_simulation, argnums=0)


if __name__ == '__main__':
    mesh = ReadExodusMesh.read_exodus_mesh(input_mesh)
    objective_value, grads = run_simulation(mesh.coords, mesh)

    ExodusWriter.copy_exodus_mesh(input_mesh, output_file)
    exo = ExodusWriter.setup_exodus_database(
        output_file,
        2, 0, ['grad_x', 'grad_y'], []
    )
    exo.put_time(1, 0.)
    ExodusWriter.write_exodus_nodal_outputs(
        exo,
        ['grad_x', 'grad_y'], [grads[:, 0], grads[:, 1]], time_step=1
    )
