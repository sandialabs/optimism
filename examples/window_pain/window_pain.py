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

from optimism import AlSolver

from optimism.contact.Contact import compute_closest_distance_to_each_side_smooth as closest_distance_func
from optimism.contact.Contact import get_potential_interaction_list

from optimism.ConstrainedObjective import ConstrainedQuasiObjective


if __name__ == '__main__':
    mesh = ReadExodusMesh.read_exodus_mesh('./geometry.g')
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
    quad_rule_face = QuadratureRule.create_quadrature_rule_1D(4)
    func_space = FunctionSpace.construct_function_space(mesh, quad_rule)

    ebcs = [
            EssentialBC(nodeSet='nset_outer_bottom', component=0),
            EssentialBC(nodeSet='nset_outer_bottom', component=1),
            EssentialBC(nodeSet='nset_outer_top', component=0),
            EssentialBC(nodeSet='nset_outer_top', component=1)
    ]
    
    contact_edges_1 = mesh.sideSets['sset_inner_left']
    contact_edges_2 = mesh.sideSets['sset_inner_right']

    dofManager = DofManager(func_space, 2, ebcs)

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

    # contact solver settings
    al_settings = AlSolver.get_settings(
        max_gmres_iters=25,
        num_initial_low_order_iterations=10,
        use_second_order_update=True,
        penalty_scaling=1.05,
        target_constraint_decrease_factor=0.5,
        tol=5e-8
    )
    # smoothing_distance = 1e-4
    search_frequency = 1
    max_contact_neighbors = 4

    h = 0.025 # approximate element size

    smoothing_distance = 1.e-1 * h

    internal_variables = mech_funcs.compute_initial_state()

    def get_ubcs(p):
        yLoc = p[0]
        V = np.zeros(mesh.coords.shape)
        index = (mesh.nodeSets['nset_outer_top'], 1)
        V = V.at[index].set(yLoc)
        return dofManager.get_bc_values(V)


    def create_field(Uu, p):
        return dofManager.create_field(Uu, get_ubcs(p))


    def energy_function(Uu, p):
        U = create_field(Uu, p)
        # internal_variables = p[1]
        return mech_funcs.compute_strain_energy(U, internal_variables)


    def energy_function_with_contact(Uu, lam, p):
        return energy_function(Uu, p)


    def constraint_function(Uu, p):
        U = create_field(Uu, p)
        interaction_list = p[1]
        contact_dist_1 = closest_distance_func(mesh, U, quad_rule_face, interaction_list[0], contact_edges_1, smoothing_distance).ravel()
        contact_dist_2 = closest_distance_func(mesh, U, quad_rule_face, interaction_list[1], contact_edges_2, smoothing_distance).ravel()

        contact_dist_1 = np.where(contact_dist_1 < -h / 8.0, 1., contact_dist_1)
        contact_dist_2 = np.where(contact_dist_2 < -h / 8.0, 1., contact_dist_2)
        return np.hstack([contact_dist_1, contact_dist_2])


    def assemble_sparse(Uu, p):
        U = create_field(Uu, p)
        internal_variables = p[1]
        element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables)
        return SparseMatrixAssembler.\
            assemble_sparse_stiffness_matrix(element_stiffnesses, func_space.mesh.conns, dofManager)


    def is_another_neighbor(qneighbor, resident):
        return np.where( np.any(qneighbor!=resident), True, False )

    
    def filter_edge_neighbors(eneighbors, resident):
        isNeighbor = jax.vmap(is_another_neighbor, (0,None))(eneighbors, resident) #[is_another_neighbor(q, resident) for q in eneighbors]
        return eneighbors[np.where(isNeighbor)]


    def update_params_function(step, Uu, p):
        # update displacement BCs
        max_disp = 0.5
        max_steps = 20
        # disp = p[0]
        # disp = disp - max_disp / max_steps
        disp = -(step / max_steps) * max_disp

        p = Objective.param_index_update(p, 0, disp)

        # update contact stuff
        if step % search_frequency == 0:
            U = create_field(Uu, p)
            interaction_list_1 = get_potential_interaction_list(contact_edges_1, contact_edges_1, mesh, U, max_contact_neighbors)
            interaction_list_1 = np.array([filter_edge_neighbors(eneighbors, contact_edges_1[e]) for e, eneighbors in enumerate(interaction_list_1)])
            interaction_list_2 = get_potential_interaction_list(contact_edges_2, contact_edges_2, mesh, U, max_contact_neighbors)
            interaction_list_2 = np.array([filter_edge_neighbors(eneighbors, contact_edges_2[e]) for e, eneighbors in enumerate(interaction_list_2)])
            interaction_lists = (interaction_list_1, interaction_list_2)
            p = Objective.param_index_update(p, 1, interaction_lists)

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


    def run_without_contact():
        Uu = dofManager.get_unknown_values(np.zeros(mesh.coords.shape))
        disp = 0.0
        ivs = mech_funcs.compute_initial_state()
        p = Objective.Params(disp, ivs)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        step = 0
        maxDisp = 0.5

        plot_solution(create_field(Uu, p), 'output-0000', p)

        steps = 20
        for step in range(1, steps):
            print('--------------------------------------')
            print('LOAD STEP ', step)
            disp = disp - maxDisp / steps

            p = Objective.param_index_update(p, 0, disp)
            Uu,_ = EqSolver.nonlinear_equation_solve(objective, Uu, p, eq_settings)
            plot_solution(create_field(Uu, p), 'output-%s' % str(step + 1).zfill(4), p)


    def run_with_contact():
    
        step = 0
        initial_multiplier = 4.0
        Uu = dofManager.get_unknown_values(np.zeros(mesh.coords.shape))
        p = update_params_function(step, Uu, Objective.Params())
        c = constraint_function(Uu, p)
        kapp0 = initial_multiplier * np.ones_like(c)
        lam0 = 1e-4 * np.abs(kapp0 * c)

        objective = ConstrainedQuasiObjective(energy_function_with_contact, constraint_function,
                                              Uu, p,
                                              lam0, kapp0)
        
        plot_solution(create_field(Uu, p), 'output-0000', p)

        for step in range(1, 20):
            print('\n------------ LOAD STEP', step, '------------\n')
                
            count=0
            def iteration_plot(Uu, p):
                nonlocal count
                # write_debug_output_func(count, Uu, p, objective.lam)
                count=count+1

            residuals=[]
            def subproblem_residual(Uu, obj):
                errorNorm = np.linalg.norm(obj.total_residual(Uu))
                residuals.append(errorNorm)
                print('error = ', errorNorm)
                with open('contact_residuals.'+str(count)+'.npz', 'wb') as file:
                    np.savez(file,
                            data=np.array(residuals))

            p = update_params_function(step, Uu, p)
            Uu = AlSolver.augmented_lagrange_solve(objective, Uu, p, al_settings, eq_settings,
                                                   callback=iteration_plot, 
                                                   sub_problem_callback=subproblem_residual)

            plot_solution(create_field(Uu, p), 'output-%s' % str(step + 1).zfill(4), p)

    run_without_contact()
    # run_with_contact()


