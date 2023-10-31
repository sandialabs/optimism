import unittest

import jax
import jax.numpy as np
from matplotlib import pyplot as plt

from optimism import EquationSolver as EqSolver
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Mechanics
from optimism import Objective
from optimism import Mesh
from optimism.material import Neohookean

from optimism.inverse.test.FiniteDifferenceFixture import FiniteDifferenceFixture

# for adjoint
import numpy as onp
from scipy.sparse import linalg

# misc. modules in test directory for now while I test
import adjoint_problem_function_space as AdjointFunctionSpace

class NeoHookeanGlobalMeshAdjointSolveFixture(FiniteDifferenceFixture):
    def setUp(self):
        dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        self.initialMesh, self.U = self.create_mesh_and_disp(4,4,[0.,1.],[0.,1.],
                                            lambda x: dispGrad0@x)

        shearModulus = 0.855 # MPa
        bulkModulus = 100*shearModulus # MPa
        youngModulus = 9.0*bulkModulus*shearModulus / (3.0*bulkModulus + shearModulus)
        poissonRatio = (3.0*bulkModulus - 2.0*shearModulus) / 2.0 / (3.0*bulkModulus + shearModulus)
        props = {
            'elastic modulus': youngModulus,
            'poisson ratio': poissonRatio,
            'version': 'coupled'
        }
        self.materialModel = Neohookean.create_material_model_functions(props)

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
            storedState.append((Uu, p))
        
        return storedState

    def strain_energy_objective(self, storedState, parameters):
        coords = parameters.reshape(self.mesh.coords.shape)
        adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
        mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)

        def energy_function_coords(Uu, p, coords):
            U = self.dofManager.create_field(Uu, p.bc_data)
            internal_vars = p[1]
            return mech_funcs.compute_strain_energy(U, internal_vars)

        return energy_function_coords(storedState[-1][0], storedState[-1][1], parameters) 

    def strain_energy_gradient(self, storedState, parameters):
        return jax.grad(self.strain_energy_objective, argnums=1)(storedState, parameters)
    
    def total_work_increment(self, Uu, Uu_prev, p, p_prev, coordinates):
        coords = coordinates.reshape(self.mesh.coords.shape)
        adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
        mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)

        def energy_function_all_dofs(U, p):
            internal_variables = p[1]
            return mech_funcs.compute_strain_energy(U, internal_variables)

        nodal_forces = jax.grad(energy_function_all_dofs, argnums=0)

        index = (self.mesh.nodeSets['left'], 1) # arbitrarily choosing left side nodeset for reaction force

        U = self.dofManager.create_field(Uu, p.bc_data)
        force = np.array(nodal_forces(U, p).at[index].get())
        disp = U.at[index].get()

        U_prev = self.dofManager.create_field(Uu_prev, p_prev.bc_data)
        force_prev = np.array(nodal_forces(U_prev, p_prev).at[index].get())
        disp_prev = U_prev.at[index].get()

        return 0.5*np.tensordot((force + force_prev),(disp - disp_prev), axes=1)

    def total_work_objective(self, storedState, parameters):
        val = 0.0
        for step in range(1, self.steps+1):
            val += self.total_work_increment(storedState[step][0], storedState[step-1][0], storedState[step][1], storedState[step-1][1], parameters)

        return val

    def total_work_gradient_just_jax(self, storedState, parameters):
        return jax.grad(self.total_work_objective, argnums=1)(storedState, parameters)

    def total_work_gradient_with_adjoint(self, storedState, parameters):
        compute_df = jax.grad(self.total_work_increment, (0, 1, 4))

        def energy_function_coords(Uu, Uu_prev, p, coordinates):
            coords = coordinates.reshape(self.mesh.coords.shape)
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
            U = self.dofManager.create_field(Uu, p.bc_data)
            internal_vars = p[1]
            return mech_funcs.compute_strain_energy(U, internal_vars)

        gradient = np.zeros(parameters.shape)
        adjointLoad = np.zeros(storedState[0][0].shape)

        for step in reversed(range(1, self.steps+1)):
            Uu = storedState[step][0]
            p = storedState[step][1]

            Uu_prev = storedState[step-1][0]
            p_prev = storedState[step-1][1]

            df_du, df_dun, df_dx = compute_df(Uu, Uu_prev, p, p_prev, parameters)

            adjointLoad -= df_du

            n = self.dofManager.get_unknown_size()
            self.objective.p = p # have to update parameters to get precond to work
            self.objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(self.objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), tol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            gradient += df_dx
            # The action dRdX * lambda (the same as lambda^T * dRdX)
            gradient += jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(Uu, Uu_prev, p, z), parameters)[1](adjointVector)[0]

            adjointLoad = -df_dun
            # The action dRdU * lambda (the same as lambda^T * dRdU) - For Hyperelastic the residual is not dependent on Uu_prev, so don't actually need this term
            # adjointLoad -= jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(Uu, z, p, parameters), Uu_prev)[1](adjointVector)[0]

        return gradient



    def test_self_adjoint_gradient(self):
        self.compute_objective_function = self.strain_energy_objective
        self.compute_gradient = self.strain_energy_gradient

        initialStepSize = 1e-5
        numSteps = 4

        errors = self.compute_finite_difference_errors(initialStepSize, numSteps, self.initialMesh.coords.ravel())

        self.assertFiniteDifferenceCheckHasVShape(errors)

    @unittest.expectedFailure
    def test_non_self_adjoint_gradient_without_adjoint_solve(self):
        self.compute_objective_function = self.total_work_objective
        self.compute_gradient = self.total_work_gradient_just_jax

        initialStepSize = 1e-5
        numSteps = 4

        errors = self.compute_finite_difference_errors(initialStepSize, numSteps, self.initialMesh.coords.ravel())

        self.assertFiniteDifferenceCheckHasVShape(errors)

    def test_non_self_adjoint_gradient_with_adjoint_solve(self):
        self.compute_objective_function = self.total_work_objective
        self.compute_gradient = self.total_work_gradient_with_adjoint

        initialStepSize = 1e-5
        numSteps = 4

        errors = self.compute_finite_difference_errors(initialStepSize, numSteps, self.initialMesh.coords.ravel())

        self.assertFiniteDifferenceCheckHasVShape(errors)



if __name__ == '__main__':
    unittest.main()