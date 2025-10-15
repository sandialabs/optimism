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
from optimism.ScipyInterface import make_scipy_linear_function

from .FiniteDifferenceFixture import FiniteDifferenceFixture

from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace
from collections import namedtuple

EnergyFunctions = namedtuple('EnergyFunctions',
                            ['energy_function_coords',
                             'compute_dissipation'])


class J2GlobalMeshAdjointSolveFixture(FiniteDifferenceFixture):
    def setUp(self):
        dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        mesh, self.U = self.create_mesh_and_disp(3,3,[0.,1.],[0.,1.],
                                            lambda x: dispGrad0@x)
        blocks = {'block0': np.array([0, 1, 2, 3]),
                  'block1': np.array([4, 5, 6, 7])}
        self.initialMesh = Mesh.mesh_with_blocks(mesh, blocks)

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
        j2_model = J2.create_material_model_functions(props)
        self.materialModels = {'block0': j2_model, 'block1': j2_model}
        self.props = {
            'block0': J2.create_material_properties(props),
            'block1': J2.create_material_properties(props)
        }

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)

        self.EBCs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]

        self.steps = 2

    def forward_solve(self, parameters):
        self.mesh = Mesh.construct_mesh_from_basic_data(parameters.reshape(self.initialMesh.coords.shape),\
                                                        self.initialMesh.conns, self.initialMesh.blocks,\
                                                        self.initialMesh.nodeSets, self.initialMesh.sideSets)

        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        mechFuncs = Mechanics.create_multi_block_mechanics_functions(functionSpace,
                                                                     "plane strain",
                                                                     self.materialModels)
        self.dofManager = FunctionSpace.DofManager(functionSpace, 2, self.EBCs)
        Ubc = self.dofManager.get_bc_values(self.U)

        Ubc_inc = Ubc / self.steps
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=np.zeros(Ubc.shape), state_data=ivs, prop_data=self.props)

        def compute_energy(Uu, p):
            U = self.dofManager.create_field(Uu, p.bc_data)
            internalVariables = p.state_data
            props = p.prop_data
            return mechFuncs.compute_strain_energy(U, internalVariables, props)

        Uu = 0.0*self.dofManager.get_unknown_values(self.U) 
        self.objective = Objective.Objective(compute_energy, Uu, p)

        storedState = []
        storedState.append((Uu, p))

        for step in range(1, self.steps+1):
            p = Objective.param_index_update(p, 0, step*Ubc_inc)
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = self.dofManager.create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data, p.prop_data)
            p = Objective.param_index_update(p, 1, ivs)
            storedState.append((Uu, p))

        return storedState

    def setup_energy_functions(self):
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quadRule.xigauss)

        def energy_function_all_dofs(U, ivs, props, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_multi_block_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModels=self.materialModels)
            return mech_funcs.compute_strain_energy(U, ivs, props)

        def energy_function_coords(Uu, p, ivs, coords):
            U = self.dofManager.create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, ivs, p.prop_data, coords)

        def dissipation(Uu, p, ivs, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_multi_block_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModels=self.materialModels)
            U = self.dofManager.create_field(Uu, p.bc_data)
            return mech_funcs.integrated_material_qoi(U, ivs, p.prop_data)

        return EnergyFunctions(energy_function_coords, jax.jit(dissipation))

    def compute_dissipated_energy(self, uSteps, pSteps, ivsSteps, coordinates, compute_dissipation):
        dissipated_energy = 0.0
        for step in range(1, self.steps+1):
            Uu = uSteps[step]
            p = pSteps[step]
            ivs = ivsSteps[step-1] # use previous step to make sure increment is correctly computed
            dissipated_energy += compute_dissipation(Uu, p, ivs, coordinates)

        return dissipated_energy

    def dissipated_energy_objective(self, storedState, parameters):
        parameters = parameters.reshape(self.mesh.coords.shape)
        energyFuncs = self.setup_energy_functions()

        uSteps = np.stack([storedState[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([storedState[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [storedState[i][1] for i in range(0, self.steps+1)]

        return self.compute_dissipated_energy(uSteps, pSteps, ivsSteps, parameters, energyFuncs.compute_dissipation) 

    def dissipated_energy_gradient(self, storedState, parameters):
        functionSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        energyFuncs = self.setup_energy_functions()
        ivsUpdateInverseFuncs = MechanicsInverse.create_ivs_update_inverse_functions_multi_block(functionSpace,
                                                                                                 "plane strain",
                                                                                                 self.materialModels)
        residualInverseFuncs = MechanicsInverse.create_path_dependent_residual_inverse_functions(energyFuncs.energy_function_coords)

        # derivatives of F
        parameters = parameters.reshape(self.mesh.coords.shape)
        uSteps = np.stack([storedState[i][0] for i in range(0, self.steps+1)], axis=0)
        ivsSteps = np.stack([storedState[i][1].state_data for i in range(0, self.steps+1)], axis=0)
        pSteps = [storedState[i][1] for i in range(0, self.steps+1)]
        df_du, df_dc, gradient = jax.grad(self.compute_dissipated_energy, (0, 2, 3))(uSteps, pSteps, ivsSteps, parameters, energyFuncs.compute_dissipation) 

        mu = np.zeros(ivsSteps[0].shape) 
        adjointLoad = np.zeros(uSteps[0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = uSteps[step]
            p = pSteps[step]
            U = self.dofManager.create_field(Uu, p.bc_data)
            ivs_prev = ivsSteps[step-1]

            dc_dcn = ivsUpdateInverseFuncs.ivs_update_jac_ivs_prev(U, ivs_prev, p.prop_data)

            mu += df_dc[step]
            adjointLoad -= df_du[step]
            adjointLoad -= self.dofManager.get_unknown_values(ivsUpdateInverseFuncs.ivs_update_jac_disp_vjp(U, ivs_prev, self.props, mu))

            n = self.dofManager.get_unknown_size()
            p_objective = Objective.Params(bc_data=p.bc_data, state_data=ivs_prev, prop_data=self.props) # remember R is a function of ivs_prev
            self.objective.p = p_objective 
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), make_scipy_linear_function(lambda V: self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), make_scipy_linear_function(self.objective.apply_precond))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), rtol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            gradient += residualInverseFuncs.residual_jac_coords_vjp(Uu, p, ivs_prev, parameters, adjointVector)
            gradient += ivsUpdateInverseFuncs.ivs_update_jac_coords_vjp(U, ivs_prev, self.props, parameters, mu)

            mu = np.einsum('ijk,ijkn->ijn', mu, dc_dcn)
            mu += residualInverseFuncs.residual_jac_ivs_prev_vjp(Uu, p, ivs_prev, parameters, adjointVector)

            adjointLoad = np.zeros(uSteps[0].shape)

        return gradient.ravel()

    def test_dissipated_energy_gradient_with_adjoint_solve(self):
        self.compute_objective_function = self.dissipated_energy_objective
        self.compute_gradient = self.dissipated_energy_gradient

        stepSize = 1e-8

        error = self.compute_finite_difference_error(stepSize, self.initialMesh.coords.ravel())
        self.assertLessEqual(error, 5e-7)


        
if __name__ == '__main__':
    unittest.main()
