import jax
import jax.numpy as np

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism.material import Neohookean as MatModel
from optimism import Mechanics
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager
from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import VTKWriter

import time


if __name__ == '__main__':
    mesh = ReadExodusMesh.read_exodus_mesh('./hole_array.exo')
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
    quad_rule_face = QuadratureRule.create_quadrature_rule_1D(4)
    func_space = FunctionSpace.construct_function_space(mesh, quad_rule)

    ebcs = [
        EssentialBC(nodeSet='yminus_nodeset', component=0),
        EssentialBC(nodeSet='yminus_nodeset', component=1),
        EssentialBC(nodeSet='yplus_nodeset', component=0),
        EssentialBC(nodeSet='yplus_nodeset', component=1)
    ]

    dofManager = DofManager(func_space, 2, ebcs)
    print(dofManager)
    props = {'elastic modulus': 3. * 10.0 * (1. - 2. * 0.3),
             'poisson ratio': 0.3,
             'version': 'coupled'}

    mat_model = MatModel.create_material_model_functions(props)
    mech_funcs = Mechanics.create_mechanics_functions(func_space, mode2D='plane strain', materialModel=mat_model)
    
    eq_settings = EqSolver.get_settings(
        use_incremental_objective=False,
        max_trust_iters=100,
        tr_size=0.25,
        min_tr_size=1e-15,
        tol=5e-8
    )

    internal_variables = mech_funcs.compute_initial_state()

    def get_ubcs(p):
        yLoc = p[0]
        V = np.zeros(mesh.coords.shape)
        index = (mesh.nodeSets['yplus_nodeset'], 1)
        V = V.at[index].set(yLoc)
        return p.dof_manager.get_bc_values(V)


    def create_field(Uu, p):
        return p.dof_manager.create_field(Uu, get_ubcs(p))


    def energy_function(Uu, p):
        U = create_field(Uu, p)
        # internal_variables = p[1]
        return mech_funcs.compute_strain_energy(U, internal_variables)


    def energy_function_with_contact(Uu, lam, p):
        return energy_function(Uu, p)


    def assemble_sparse(Uu, p):
        U = create_field(Uu, p)
        internal_variables = p[1]
        element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables)
        return SparseMatrixAssembler.\
            assemble_sparse_stiffness_matrix(element_stiffnesses, func_space.mesh.conns, dofManager)


    def update_params_function(step, Uu, p):
        # update displacement BCs
        max_disp = 5.
        max_steps = 20
        # disp = p[0]
        # disp = disp - max_disp / max_steps
        disp = -(step / max_steps) * max_disp

        p = Objective.param_index_update(p, 0, disp)

        # update contact stuff
        # if step % search_frequency == 0:
        #     U = create_field(Uu, p)
        #     interaction_list_1 = get_potential_interaction_list(contact_edges_1, contact_edges_1, mesh, U, max_contact_neighbors)
        #     interaction_list_1 = np.array([filter_edge_neighbors(eneighbors, contact_edges_1[e]) for e, eneighbors in enumerate(interaction_list_1)])
        #     interaction_list_2 = get_potential_interaction_list(contact_edges_2, contact_edges_2, mesh, U, max_contact_neighbors)
        #     interaction_list_2 = np.array([filter_edge_neighbors(eneighbors, contact_edges_2[e]) for e, eneighbors in enumerate(interaction_list_2)])
        #     interaction_lists = (interaction_list_1, interaction_list_2)
        #     p = Objective.param_index_update(p, 1, interaction_lists)

        return p


    def plot_solution(dispField, plotName, p):
        writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=dispField,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        
        bcs = np.array(dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs',
                               nodalData=bcs,
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)
        
        writer.write()


    def run():
        Uu = dofManager.get_unknown_values(np.zeros(mesh.coords.shape))
        disp = 0.0
        ivs = mech_funcs.compute_initial_state()
        p = Objective.Params(disp, ivs, dof_manager=dofManager)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        step = 0
        maxDisp = 5.0

        plot_solution(create_field(Uu, p), 'output-0000', p)

        steps = 20
        for step in range(1, steps):
            print('--------------------------------------')
            print('LOAD STEP ', step)
            disp = disp - maxDisp / steps

            p = Objective.param_index_update(p, 0, disp)
            Uu,_ = EqSolver.nonlinear_equation_solve(objective, Uu, p, eq_settings)
            plot_solution(create_field(Uu, p), 'output-%s' % str(step + 1).zfill(4), p)

    # run_without_contact()

if __name__ == '__main__':
    times = []
    for n in range(10):
        start_time = time.time()
        run()
        total_time = time.time() - start_time
        print(f'  Sim {n + 1} time = {total_time}')
        times.append(total_time)
    print(f'Average time = {sum(times) / len(times)}')