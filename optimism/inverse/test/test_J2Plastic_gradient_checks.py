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

from optimism.inverse.test.FiniteDifferenceFixture import FiniteDifferenceFixture

from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace

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

    def total_work_increment(self, Uu, Uu_prev, ivs, ivs_prev, p, p_prev, coordinates, nodal_forces):
        index = (self.mesh.nodeSets['left'], 1) # arbitrarily choosing left side nodeset for reaction force

        U = self.dofManager.create_field(Uu, p.bc_data)
        force = np.array(nodal_forces(U, ivs, coordinates).at[index].get())
        disp = U.at[index].get()

        U_prev = self.dofManager.create_field(Uu_prev, p_prev.bc_data)
        force_prev = np.array(nodal_forces(U_prev, ivs_prev, coordinates).at[index].get())
        disp_prev = U_prev.at[index].get()

        return 0.5*np.tensordot((force + force_prev),(disp - disp_prev), axes=1)

    def total_work_objective(self, storedState, parameters):
        parameters = parameters.reshape(self.mesh.coords.shape)

        def energy_function_all_dofs(U, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain',materialModel=self.materialModel)
            return mech_funcs.compute_strain_energy(U, ivs)

        nodal_forces = jax.jit(jax.grad(energy_function_all_dofs, argnums=0))

        val = 0.0
        for step in range(1, self.steps+1):
            Uu = storedState[step][0]
            Uu_prev = storedState[step-1][0]
            p = storedState[step][1]
            p_prev = storedState[step-1][1]
            ivs = p.state_data
            ivs_prev = p_prev.state_data

            val += self.total_work_increment(Uu, Uu_prev, ivs, ivs_prev, p, p_prev, parameters, nodal_forces)

        return val

    def total_work_gradient(self, storedState, parameters):

        def energy_function_coords(Uu, p, ivs_prev, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
            U = self.dofManager.create_field(Uu, p.bc_data)
            return mech_funcs.compute_strain_energy(U, ivs_prev)

        def energy_function_all_dofs(U, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain',materialModel=self.materialModel)
            return mech_funcs.compute_strain_energy(U, ivs)

        nodal_forces = jax.jit(jax.grad(energy_function_all_dofs, argnums=0))

        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        ivsUpdateInverseFuncs = MechanicsInverse.create_ivs_update_inverse_functions(functionSpace,
                                                                                     "plane strain",
                                                                                     self.materialModel)

        residualInverseFuncs = MechanicsInverse.create_path_dependent_residual_inverse_functions(energy_function_coords)

        compute_df = jax.grad(self.total_work_increment, (0, 1, 2, 3, 6))

        parameters = parameters.reshape(self.mesh.coords.shape)
        gradient = np.zeros(parameters.shape)
        mu = np.zeros(storedState[0][1].state_data.shape) 
        adjointLoad = np.zeros(storedState[0][0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = storedState[step][0]
            Uu_prev = storedState[step-1][0]
            p = storedState[step][1]
            p_prev = storedState[step-1][1]
            ivs = p.state_data
            ivs_prev = p_prev.state_data

            dc_dcn = ivsUpdateInverseFuncs.ivs_update_jac_ivs_prev(self.dofManager.create_field(Uu, p.bc_data), ivs_prev)

            df_du, df_dun, df_dc, df_dcn, df_dx = compute_df(Uu, Uu_prev, ivs, ivs_prev, p, p_prev, parameters, nodal_forces)

            mu += df_dc
            adjointLoad -= df_du
            adjointLoad -= self.dofManager.get_unknown_values(ivsUpdateInverseFuncs.ivs_update_jac_disp_vjp(self.dofManager.create_field(Uu, p.bc_data), ivs_prev, mu))

            n = self.dofManager.get_unknown_size()
            p_objective = Objective.Params(bc_data=p.bc_data, state_data=p_prev.state_data) # remember R is a function of ivs_prev
            self.objective.p = p_objective 
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), tol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            gradient += df_dx
            gradient += residualInverseFuncs.residual_jac_coords_vjp(Uu, p, ivs_prev, parameters, adjointVector)
            gradient += ivsUpdateInverseFuncs.ivs_update_jac_coords_vjp(self.dofManager.create_field(Uu, p.bc_data), ivs_prev, parameters, mu)

            mu = np.einsum('ijk,ijkn->ijn', mu, dc_dcn)
            mu += residualInverseFuncs.residual_jac_ivs_prev_vjp(Uu, p, ivs_prev, parameters, adjointVector)
            mu += df_dcn 

            adjointLoad = -df_dun

        return gradient.ravel()

    def test_gradient_with_adjoint_solve(self):
        self.compute_objective_function = self.total_work_objective
        self.compute_gradient = self.total_work_gradient

        initialStepSize = 1e-5
        numSteps = 4

        errors = self.compute_finite_difference_errors(initialStepSize, numSteps, self.initialMesh.coords.ravel())
        self.assertFiniteDifferenceCheckHasVShape(errors)


        
if __name__ == '__main__':
    unittest.main()
