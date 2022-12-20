from optimism.JaxConfig import *

from optimism import EquationSolver
from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import VTKWriter
from optimism.FunctionSpace import DofManager
from optimism.FunctionSpace import FunctionSpace
from optimism.Mechanics import MechanicsFunctions
from optimism.Mesh import Mesh

from optimism.helper_methods.Postprocessor import write_standard_fields_quasi_static

from typing import List


class ObjectiveError(Exception): pass
class BCTypeError(Exception): pass


def run_quasi_static_mechanics_simulation(
    mesh: Mesh, 
    f_space: FunctionSpace,
    dof_manager: DofManager, 
    mech_funcs: MechanicsFunctions,
    bcs: List[dict],
    objective_type: str,
    solver_settings: dict,
    time_settings: dict,
    pp_settings: dict) -> None:

    print('Running quasi-static mechanics simulation')

    # setup displacement bcs
    disp_bcs = bcs['displacement']
    disp_bcs_node_sets = []
    disp_bcs_components = []
    disp_bcs_values = []
    for disp_bc in disp_bcs:
        if disp_bc['type'] == 'fixed':
            continue
        elif disp_bc['type'] == 'prescribed':
            print('prescribed BC')
            disp_bcs_node_sets.append(disp_bc['nodeset'])
            disp_bcs_components.append(disp_bc['component'])
            disp_bcs_values.append(disp_bc['max displacement'])
        else:
            print('Unsuported displacement bc type "%s"' % disp_bc['type'])
            raise BCTypeError

    def get_ubcs(p):
        V = np.zeros_like(mesh.coords)
        applied_disps = p[0]
        for n in range(len(disp_bcs_node_sets)):
            ebc_indices = (mesh.nodeSets[disp_bcs_node_sets[n]], disp_bcs_components[n])
            V = V.at[ebc_indices].set(applied_disps[n])
        return dof_manager.get_bc_values(V)

    def create_field(Uu, p):
        return dof_manager.create_field(Uu, get_ubcs(p))

    def energy_function(Uu, p):
        U = create_field(Uu, p)
        internal_variables = p.state_data
        return mech_funcs.compute_strain_energy(U, internal_variables)

    def assemble_sparse_preconditioner_matrix(Uu, p):
        U = create_field(Uu, p)
        internal_variables = p.state_data
        el_stifnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(el_stifnesses, mesh.conns, dof_manager)


    # solver setup
    #
    solver_settings = EquationSolver.get_settings(
        max_cumulative_cg_iters=solver_settings['max cumulative cg iters'],
        max_trust_iters=solver_settings['max trust iters'],
        use_preconditioned_inner_product_for_cg=solver_settings['use preconditioned inner product for cg']
    )
    precond_strategy = Objective.PrecondStrategy(assemble_sparse_preconditioner_matrix)

    # unknown and params setup
    Uu = np.zeros(dof_manager.get_unknown_size())
    p = Objective.Params(np.array(len(disp_bcs_values) * [0.0]), mech_funcs.compute_initial_state())

    if objective_type.lower() == 'scaled objective':
        objective = Objective.ScaledObjective(energy_function, Uu, p, precondStrategy=precond_strategy)
    elif objective_type.lower() == 'objective':
        objective = Objective.Objective(energy_function, Uu, p, precondStrategy=precond_strategy)
    else:
        print('Unsupported objective type "%s".' % objective_type)

    # TODO parameterize this later
    # p = Objective.param_index_update(p, 0, 0.3)
    # p = Objective.param_index_update(p, 0, np.array(disp_bcs_values))
    
    # solve
    # Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, solver_settings)


    # setup
    step = 0
    disps = np.array(len(disp_bcs_values) * [0.0])
    # write initial output
    if pp_settings['type'] == 'vtk':
        write_standard_fields_quasi_static(pp_settings['output file base name'], 0, Uu, p, f_space, create_field, mech_funcs)
    else:
        raise ValueError('Unsupported post processor')

    # now loop
    for step in range(1, time_settings['number of steps']):
        print('====================================================')
        print('==== LOAD STEP = %s' % step)
        print('====================================================')

        # update bc stuff
        for n in range(len(disps)):
            disps = disps.at[n].set(disps[n] + disp_bcs_values[n] / time_settings['number of steps'])
        
        p = Objective.param_index_update(p, 0, disps)
        Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, solver_settings)

        if pp_settings['type'] == 'vtk':
            write_standard_fields_quasi_static(pp_settings['output file base name'], step, Uu, p, f_space, create_field, mech_funcs)
        else:
            raise ValueError('Unsupported post processor')


