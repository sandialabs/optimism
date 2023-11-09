import unittest

import jax
import jax.numpy as np
import numpy as onp
from scipy.sparse import linalg

from optimism import EquationSolver as EqSolver
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Mechanics
from optimism import Objective
from optimism import Mesh
from optimism.material import J2Plastic as J2

from optimism.Timer import Timer

from optimism.inverse.test.FiniteDifferenceFixture import FiniteDifferenceFixture

# misc. modules in test directory for now while I test
# from optimism.inverse import MechanicsInverse
import MechanicsInverse
import adjoint_problem_function_space as AdjointFunctionSpace

class J2GlobalMeshAdjointSolveFixture(FiniteDifferenceFixture):
    def setUp(self):
        dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        self.initialMesh, self.U = self.create_mesh_and_disp(4,4,[0.,1.],[0.,1.],
                                            lambda x: dispGrad0@x)

        E = 100.0
        poisson = 0.321
        H = 1e-2 * E
        Y0 = 0.3 * E
        props = {
            'elastic modulus': E,
            'poisson ratio': poisson,
            'yield strength': Y0,
            'kinematics': 'small deformations',
            # 'kinematics': 'large deformations',
            'hardening model': 'linear',
            'hardening modulus': H
        }
        self.materialModel = J2.create_material_model_functions(props)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)

        self.EBCs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]

        self.steps = 2

    def forward_solve(self, parameters):
        self.mesh = Mesh.construct_mesh_from_basic_data(parameters.reshape(self.initialMesh.coords.shape),\
                                                        self.initialMesh.conns, self.initialMesh.blocks,\
                                                        self.initialMesh.nodeSets, self.initialMesh.sideSets)

        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        mechFuncs = Mechanics.create_mechanics_functions(functionSpace,
                                                         "plane strain",
                                                         self.materialModel)
        self.dofManager = FunctionSpace.DofManager(functionSpace, 2, self.EBCs)
        Ubc = self.dofManager.get_bc_values(self.U)

        Ubc_inc = Ubc / self.steps
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=np.zeros(Ubc.shape), state_data=ivs)

        def compute_energy(Uu, p):
            U = self.dofManager.create_field(Uu, p.bc_data)
            internalVariables = p.state_data
            return mechFuncs.compute_strain_energy(U, internalVariables)

        Uu = 0.0*self.dofManager.get_unknown_values(self.U) 
        self.objective = Objective.Objective(compute_energy, Uu, p)

        storedState = []
        storedState.append((Uu, p))

        for step in range(1, self.steps+1):
            p = Objective.param_index_update(p, 0, step*Ubc_inc)
            Uu = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = self.dofManager.create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data)
            p = Objective.param_index_update(p, 1, ivs)
            storedState.append((Uu, p))

        return storedState

    def strain_energy_increment(self, Uu, ivs, p, coordinates):
        coords = coordinates.reshape(self.mesh.coords.shape)
        internal_vars = ivs.reshape(p.state_data.shape)
        adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
        mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
        U = self.dofManager.create_field(Uu, p.bc_data)

        return mech_funcs.compute_strain_energy(U, internal_vars)

    def strain_energy_objective(self, storedState, parameters):
        val = 0.0
        for step in range(1, self.steps+1):
            Uu = storedState[step][0]
            p = storedState[step][1]
            ivs = storedState[step][1].state_data.ravel()
            val += self.strain_energy_increment(Uu, ivs, p, parameters)

        return val

    def strain_energy_gradient(self, storedState, parameters):
        return jax.grad(self.strain_energy_objective, argnums=1)(storedState, parameters)

    # def total_work_increment(self, Uu, Uu_prev, ivs, ivs_prev, p, p_prev, coordinates):
    #     coords = coordinates.reshape(self.mesh.coords.shape)
    #     internal_vars = ivs.reshape(p.state_data.shape)
    #     internal_vars_prev = ivs_prev.reshape(p_prev.state_data.shape)

    #     adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
    #     mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)

    #     def energy_function_all_dofs(U, ivs):
    #         return mech_funcs.compute_strain_energy(U, ivs)

    #     nodal_forces = jax.grad(energy_function_all_dofs, argnums=0)

    #     index = (self.mesh.nodeSets['left'], 1) # arbitrarily choosing left side nodeset for reaction force

    #     U = self.dofManager.create_field(Uu, p.bc_data)
    #     force = np.array(nodal_forces(U, internal_vars).at[index].get())
    #     # force = np.array(self.nodal_forces(U, internal_vars).at[index].get())
    #     disp = U.at[index].get()

    #     U_prev = self.dofManager.create_field(Uu_prev, p_prev.bc_data)
    #     force_prev = np.array(nodal_forces(U_prev, internal_vars_prev).at[index].get())
    #     # force_prev = np.array(self.nodal_forces(U_prev, internal_vars_prev).at[index].get())
    #     disp_prev = U_prev.at[index].get()

    #     return 0.5*np.tensordot((force + force_prev),(disp - disp_prev), axes=1)

    def total_work_increment(self, Uu, Uu_prev, ivs, ivs_prev, p, p_prev, coordinates, nodal_forces):
        internal_vars = ivs.reshape(p.state_data.shape)
        internal_vars_prev = ivs_prev.reshape(p_prev.state_data.shape)

        index = (self.mesh.nodeSets['left'], 1) # arbitrarily choosing left side nodeset for reaction force

        U = self.dofManager.create_field(Uu, p.bc_data)
        force = np.array(nodal_forces(U, internal_vars, coordinates).at[index].get())
        disp = U.at[index].get()

        U_prev = self.dofManager.create_field(Uu_prev, p_prev.bc_data)
        force_prev = np.array(nodal_forces(U_prev, internal_vars_prev, coordinates).at[index].get())
        disp_prev = U_prev.at[index].get()

        return 0.5*np.tensordot((force + force_prev),(disp - disp_prev), axes=1)

    def total_work_objective(self, storedState, parameters):
        # functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        # adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
        # mech_funcs = Mechanics.create_mechanics_functions(functionSpace, mode2D='plane strain', materialModel=self.materialModel)
        # self.nodal_forces = jax.jit(jax.grad(mech_funcs.compute_strain_energy, argnums=0))

        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        def createField(Uu, p):
            return self.dofManager.create_field(Uu, p.bc_data)
        mechInverseFuncs = MechanicsInverse.create_mechanics_inverse_functions(functionSpace,
                                                                              createField,
                                                                              "plane strain",
                                                                              self.materialModel)

        val = 0.0
        for step in range(1, self.steps+1):
            Uu = storedState[step][0]
            Uu_prev = storedState[step-1][0]
            p = storedState[step][1]
            p_prev = storedState[step-1][1]
            ivs = p.state_data.ravel()
            ivs_prev = p_prev.state_data.ravel()

            with Timer(name="total_work_increment"):
                val += self.total_work_increment(Uu, Uu_prev, ivs, ivs_prev, p, p_prev, parameters, mechInverseFuncs.nodal_forces_parameterized)

        return val

    def total_work_gradient_with_adjoint(self, storedState, parameters):

        # def energy_function_coords(Uu, ivs_prev, p, coordinates):
        #     coords = coordinates.reshape(self.mesh.coords.shape)
        #     internal_vars = ivs_prev.reshape(p.state_data.shape)
        #     adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
        #     mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
        #     U = self.dofManager.create_field(Uu, p.bc_data)
        #     return mech_funcs.compute_strain_energy(U, internal_vars)

        # def update_internal_vars_test(Uu, ivs_prev, p, coordinates):
        #     coords = coordinates.reshape(self.mesh.coords.shape)
        #     internal_vars = ivs_prev.reshape(p.state_data.shape)
        #     adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
        #     mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
        #     U = self.dofManager.create_field(Uu, p.bc_data)
        #     return mech_funcs.compute_updated_internal_variables(U, internal_vars).ravel()

        compute_df = jax.grad(self.total_work_increment, (0, 1, 2, 3, 6))

        # Here I go trying some stuff out
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(parameters.reshape(self.mesh.coords.shape), self.mesh, self.quadRule)
        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        def createField(Uu, p):
            return self.dofManager.create_field(Uu, p.bc_data)
        mechInverseFuncs = MechanicsInverse.create_mechanics_inverse_functions(functionSpace,
                                                                              createField,
                                                                              "plane strain",
                                                                              self.materialModel)

        # ivs_update_jac_disp = jax.jit(lambda x, y, q, r, vx:
        #                               jax.vjp(lambda z: update_internal_vars_test(z, y, q, r), x)[1](vx))

        # dr_dcn_vjp = jax.jit(lambda u, iv, q, x, vx:
        #                      jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(u, z, q, x), iv)[1](vx))

        # dr_dx_vjp = jax.jit(lambda u, iv, q, x, vx:
        #                      jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(u, iv, q, z), x)[1](vx))

        # compute_dc = jax.jacfwd(update_internal_vars_test, (0, 1, 3))
        # compute_dc = jax.jacfwd(update_internal_vars_test, (0, 3))
        # compute_dc = jax.jacfwd(update_internal_vars_test, 3)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        gradient = np.zeros(parameters.shape)
        mu = np.zeros(np.prod(np.array(storedState[0][1].state_data.shape))) 
        adjointLoad = np.zeros(storedState[0][0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = storedState[step][0]
            Uu_prev = storedState[step-1][0]
            p = storedState[step][1]
            p_prev = storedState[step-1][1]
            ivs = p.state_data.ravel()
            ivs_prev = p_prev.state_data.ravel()

            with Timer(name="compute_dc"):
                # dc_du, dc_dcn, dc_dx = compute_dc(Uu, ivs_prev, p, parameters)
                # dc_du, dc_dx = compute_dc(Uu, ivs_prev, p, parameters)
                # dc_dx = compute_dc(Uu, ivs_prev, p, parameters)
                dc_dcn = mechInverseFuncs.ivs_update_jac_ivs_prev(self.dofManager.create_field(Uu, p.bc_data), ivs_prev.reshape(p_prev.state_data.shape))
                # dc_dx = mechInverseFuncs.ivs_update_jac_coords(self.dofManager.create_field(Uu, p.bc_data), ivs_prev.reshape(p_prev.state_data.shape), parameters)

            with Timer(name="compute_df"):
                # df_du, df_dun, df_dc, df_dcn, df_dx = compute_df(Uu, Uu_prev, ivs, ivs_prev, p, p_prev, parameters)
                df_du, df_dun, df_dc, df_dcn, df_dx = compute_df(Uu, Uu_prev, ivs, ivs_prev, p, p_prev, parameters, mechInverseFuncs.nodal_forces_parameterized)

            with Timer(name="first mu and adjoint load updates"):
                mu += df_dc
                adjointLoad -= df_du

                # adjointLoad -= np.tensordot(mu, dc_du, axes=1) # mu^T dc/du
                # adjointLoad -= ivs_update_jac_disp(Uu, ivs_prev, p, parameters, mu)[0]
                adjointLoad -= self.dofManager.get_unknown_values(mechInverseFuncs.ivs_update_jac_disp_vjp(self.dofManager.create_field(Uu, p.bc_data), ivs_prev.reshape(p_prev.state_data.shape), mu.reshape(p.state_data.shape)))

            with Timer(name="adjoint solve"):
                n = self.dofManager.get_unknown_size()
                p_objective = Objective.Params(bc_data=p.bc_data, state_data=p_prev.state_data) # remember R is a function of ivs_prev
                self.objective.p = p_objective 
                self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
                dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
                dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
                adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), tol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            with Timer(name="update gradient"):
                gradient += df_dx
                # The action dRdX * lambda (the same as lambda^T * dRdX)
                # gradient += jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(Uu, ivs_prev, p, z), parameters)[1](adjointVector)[0]
                # gradient += dr_dx_vjp(Uu, ivs_prev, p, parameters, adjointVector)[0]
                gradient += mechInverseFuncs.residual_jac_coords_vjp(Uu, p, ivs_prev.reshape(p_prev.state_data.shape), parameters, adjointVector).ravel()

                # gradient += np.tensordot(mu, dc_dx, axes=1) # mu^T dc/dx
                # gradient += np.einsum('ijk,ijkn->n', mu.reshape(p.state_data.shape), dc_dx)
                gradient += mechInverseFuncs.ivs_update_jac_coords_vjp(self.dofManager.create_field(Uu, p.bc_data), ivs_prev.reshape(p_prev.state_data.shape), parameters, mu.reshape(p.state_data.shape))

            with Timer(name="update mu"):

                # mu = np.tensordot(mu, dc_dcn, axes=1)
                mu = np.einsum('ijk,ijkn->ijn', mu.reshape(p.state_data.shape), dc_dcn).ravel()

                # The action dRdcn * lambda (the same as lambda^T * dRdcn) - Is the residual dependent on ivs_prev? Is this term needed?
                # mu += jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(Uu, z, p, parameters), ivs_prev)[1](adjointVector)[0]
                # mu += dr_dcn_vjp(Uu, ivs_prev, p, parameters, adjointVector)[0]
                mu += mechInverseFuncs.residual_jac_ivs_prev_vjp(Uu, p, ivs_prev.reshape(p_prev.state_data.shape), parameters, adjointVector).ravel()

                mu += df_dcn 

            with Timer(name="update adjoint load"):
                adjointLoad = -df_dun

        return gradient


    def test_gradient_with_adjoint_solve(self):

        # self.compute_objective_function = self.strain_energy_objective
        # self.compute_gradient = self.strain_energy_gradient


        self.compute_objective_function = self.total_work_objective
        self.compute_gradient = self.total_work_gradient_with_adjoint

        initialStepSize = 1e-5
        numSteps = 4

        errors = self.compute_finite_difference_errors(initialStepSize, numSteps, self.initialMesh.coords.ravel())

        self.assertFiniteDifferenceCheckHasVShape(errors)


        
if __name__ == '__main__':
    unittest.main()
