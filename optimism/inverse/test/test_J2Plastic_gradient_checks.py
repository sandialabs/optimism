import unittest

import jax
import jax.numpy as np
import numpy as onp
from scipy.sparse import linalg

from optimism import EquationSolver as EqSolver
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mechanics
from optimism import Objective
from optimism import Mesh
from optimism.material import J2Plastic as J2

from .FiniteDifferenceFixture import FiniteDifferenceFixture

from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace
from collections import namedtuple

EnergyFunctions = namedtuple('EnergyFunctions',
                            ['energy_function_coords',
                             'nodal_forces'])

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
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = self.dofManager.create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data)
            p = Objective.param_index_update(p, 1, ivs)
            storedState.append((Uu, p))

        return storedState

    def setup_energy_functions(self):
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quadRule.xigauss)

        def energy_function_all_dofs(U, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
            return mech_funcs.compute_strain_energy(U, ivs)

        def energy_function_coords(Uu, p, ivs, coords):
            U = self.dofManager.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, ivs, coords)

        nodal_forces = jax.grad(energy_function_all_dofs, argnums=0)

        return EnergyFunctions(energy_function_coords, jax.jit(nodal_forces))

    def compute_total_work(self, uSteps, pSteps, ivsSteps, coordinates, nodal_forces):
        index = (self.mesh.nodeSets['left'], 1) # arbitrarily choosing left side nodeset for reaction force

        totalWork = 0.0
        for step in range(1, self.steps+1):
            Uu = uSteps[step]
            p = pSteps[step]
            ivs = ivsSteps[step] 
            U = self.dofManager.create_field(Uu, p.bc_data)
            force = np.array(nodal_forces(U, ivs, coordinates).at[index].get())
            disp = U.at[index].get()

            Uu_prev = uSteps[step-1]
            p_prev = pSteps[step-1]
            ivs_prev = ivsSteps[step-1] 
            U_prev = self.dofManager.create_field(Uu_prev, p_prev.bc_data)
            force_prev = np.array(nodal_forces(U_prev, ivs_prev, coordinates).at[index].get())
            disp_prev = U_prev.at[index].get()

            totalWork += 0.5*np.tensordot((force + force_prev),(disp - disp_prev), axes=1)

        return totalWork

    def total_work_objective(self, storedState, parameters):
        parameters = parameters.reshape(self.mesh.coords.shape)
        energyFuncs = self.setup_energy_functions()

        uSteps = np.stack([storedState[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([storedState[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [storedState[i][1] for i in range(0, self.steps+1)]

        return self.compute_total_work(uSteps, pSteps, ivsSteps, parameters, energyFuncs.nodal_forces) 

    def total_work_gradient(self, storedState, parameters):
        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        energyFuncs = self.setup_energy_functions()
        ivsUpdateInverseFuncs = MechanicsInverse.create_ivs_update_inverse_functions(functionSpace,
                                                                                    "plane strain",
                                                                                    self.materialModel)
        residualInverseFuncs = MechanicsInverse.create_path_dependent_residual_inverse_functions(energyFuncs.energy_function_coords)

        # derivatives of F
        parameters = parameters.reshape(self.mesh.coords.shape)
        uSteps = np.stack([storedState[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([storedState[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [storedState[i][1] for i in range(0, self.steps+1)]
        df_du, df_dc, gradient = jax.grad(self.compute_total_work, (0, 2, 3))(uSteps, pSteps, ivsSteps, parameters, energyFuncs.nodal_forces) 

        mu = np.zeros(ivsSteps[0].shape) 
        adjointLoad = np.zeros(uSteps[0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = uSteps[step]
            p = pSteps[step]
            U = self.dofManager.create_field(Uu, p.bc_data)
            ivs_prev = ivsSteps[step-1]

            dc_dcn = ivsUpdateInverseFuncs.ivs_update_jac_ivs_prev(U, ivs_prev)

            mu += df_dc[step]
            adjointLoad -= df_du[step]
            adjointLoad -= self.dofManager.get_unknown_values(ivsUpdateInverseFuncs.ivs_update_jac_disp_vjp(U, ivs_prev, mu))

            n = self.dofManager.get_unknown_size()
            p_objective = Objective.Params(bc_data=p.bc_data, state_data=ivs_prev) # remember R is a function of ivs_prev
            self.objective.p = p_objective 
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), rtol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            gradient += residualInverseFuncs.residual_jac_coords_vjp(Uu, p, ivs_prev, parameters, adjointVector)
            gradient += ivsUpdateInverseFuncs.ivs_update_jac_coords_vjp(U, ivs_prev, parameters, mu)

            mu = np.einsum('ijk,ijkn->ijn', mu, dc_dcn)
            mu += residualInverseFuncs.residual_jac_ivs_prev_vjp(Uu, p, ivs_prev, parameters, adjointVector)

            adjointLoad = np.zeros(uSteps[0].shape)

        return gradient.ravel()

    def compute_L2_norm_difference(self, uSteps, ivsSteps, bcsSteps, coordinates, nodal_forces):
        index = (self.mesh.nodeSets['left'], 1) # arbitrarily choosing left side nodeset for reaction force

        numerator = 0.0
        denominator= 0.0
        for i in range(0, len(self.targetSteps)):
            step = self.targetSteps[i]
            Uu = uSteps[step]
            bc_data = bcsSteps[step]
            ivs = ivsSteps[step]

            U = self.dofManager.create_field(Uu, bc_data)
            force = np.sum(np.array(nodal_forces(U, ivs, coordinates).at[index].get()))

            diff = force - self.targetForces[i]
            numerator += diff*diff
            denominator += self.targetForces[i]*self.targetForces[i]

        return np.sqrt(numerator/denominator)

    def target_curve_objective(self, storedState, parameters):
        parameters = parameters.reshape(self.mesh.coords.shape)
        energyFuncs = self.setup_energy_functions()

        uSteps = np.stack([storedState[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([storedState[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        bcsSteps = np.stack([storedState[i][1].bc_data for i in range(0, self.steps+1)], axis=0)

        return self.compute_L2_norm_difference(uSteps, ivsSteps, bcsSteps, parameters, energyFuncs.nodal_forces) 

    def target_curve_gradient(self, storedState, parameters):
        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        energyFuncs = self.setup_energy_functions()
        ivsUpdateInverseFuncs = MechanicsInverse.create_ivs_update_inverse_functions(functionSpace,
                                                                                     "plane strain",
                                                                                     self.materialModel)

        residualInverseFuncs = MechanicsInverse.create_path_dependent_residual_inverse_functions(energyFuncs.energy_function_coords)

        parameters = parameters.reshape(self.mesh.coords.shape)

        # derivatives of F
        uSteps = np.stack([storedState[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([storedState[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        bcsSteps = np.stack([storedState[i][1].bc_data for i in range(0, self.steps+1)], axis=0)
        df_du, df_dc, gradient = jax.grad(self.compute_L2_norm_difference, (0, 1, 3))(uSteps, ivsSteps, bcsSteps, parameters, energyFuncs.nodal_forces) 

        mu = np.zeros(ivsSteps[0].shape) 
        adjointLoad = np.zeros(uSteps[0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = uSteps[step]
            p = storedState[step][1]
            p_prev = storedState[step-1][1]
            ivs_prev = ivsSteps[step-1]

            dc_dcn = ivsUpdateInverseFuncs.ivs_update_jac_ivs_prev(self.dofManager.create_field(Uu, p.bc_data), ivs_prev)

            mu += df_dc[step]
            adjointLoad -= df_du[step]
            adjointLoad -= self.dofManager.get_unknown_values(ivsUpdateInverseFuncs.ivs_update_jac_disp_vjp(self.dofManager.create_field(Uu, p.bc_data), ivs_prev, mu))

            n = self.dofManager.get_unknown_size()
            p_objective = Objective.Params(bc_data=p.bc_data, state_data=p_prev.state_data) # remember R is a function of ivs_prev
            self.objective.p = p_objective 
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), rtol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            gradient += residualInverseFuncs.residual_jac_coords_vjp(Uu, p, ivs_prev, parameters, adjointVector)
            gradient += ivsUpdateInverseFuncs.ivs_update_jac_coords_vjp(self.dofManager.create_field(Uu, p.bc_data), ivs_prev, parameters, mu)

            mu = np.einsum('ijk,ijkn->ijn', mu, dc_dcn)
            mu += residualInverseFuncs.residual_jac_ivs_prev_vjp(Uu, p, ivs_prev, parameters, adjointVector)

            adjointLoad = np.zeros(storedState[0][0].shape)

        return gradient.ravel()


    def test_total_work_gradient_with_adjoint_solve(self):
        self.compute_objective_function = self.total_work_objective
        self.compute_gradient = self.total_work_gradient

        stepSize = 1e-7

        error = self.compute_finite_difference_error(stepSize, self.initialMesh.coords.ravel())
        self.assertLessEqual(error, 1e-7)

    def test_target_curve_gradient_with_adjoint_solve(self):
        self.compute_objective_function = self.target_curve_objective
        self.compute_gradient = self.target_curve_gradient

        self.targetSteps = [1, 2]
        self.targetForces = [4.5, 5.5] # [4.542013626078756, 5.7673988583067555] actual forces

        stepSize = 1e-8

        error = self.compute_finite_difference_error(stepSize, self.initialMesh.coords.ravel())
        self.assertLessEqual(error, 1e-6)


        
if __name__ == '__main__':
    unittest.main()
