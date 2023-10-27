import unittest

import jax
import jax.numpy as np
from jax.scipy import linalg
from matplotlib import pyplot as plt

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism.material import J2Plastic as J2
from optimism.material import MaterialUniaxialSimulator
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism.test.TestFixture import TestFixture
from optimism.test.MeshFixture import MeshFixture

from optimism import TensorMath

# for adjoint
import numpy as onp
from scipy.sparse import linalg

# misc. modules in test directory for now while I test
import adjoint_problem_function_space as AdjointFunctionSpace


plotting=False


def make_disp_grad_from_strain(strain):
    return linalg.expm(strain) - np.identity(3)
        
def small_strain_linear_hardening_analytic_gradients(disp_grad, state, state_previous, props):
    sqrt32 = np.sqrt(3./2.)
    mu = 0.5*props['elastic modulus']/(1.0 + props['poisson ratio'])
    c0 = 3.*mu + props['hardening modulus']

    vol_projection = np.zeros((9,9)).at[(0,0,0,4,4,4,8,8,8),(0,4,8,0,4,8,0,4,8)].set(1.)
    dev_projection = np.eye(9,9) - vol_projection/3.
    sym_projection = np.zeros((9,9)).at[(1,1,2,2,3,3,5,5,6,6,7,7),(1,3,2,6,1,3,5,7,2,6,5,7)].set(0.5).at[(0,4,8),(0,4,8)].set(1.)

    eqps = state[0]
    deqps = eqps - state_previous[0]

    eps_pn = state_previous[1:]
    eps = TensorMath.sym(disp_grad)
    eps_e_tr = eps - eps_pn.reshape((3,3))

    dev_eps_e_tr = TensorMath.dev(eps_e_tr)
    dev_eps_e_tr_squared = np.tensordot(dev_eps_e_tr, dev_eps_e_tr)
    n_tr = 1./np.sqrt(dev_eps_e_tr_squared) * dev_eps_e_tr

    deqps_deps = (2.*mu * sqrt32/ c0)*n_tr
    deps_p_deps = (sqrt32 * deqps / np.sqrt(dev_eps_e_tr_squared)) * (np.tensordot(dev_projection,sym_projection,axes=1) - np.kron(n_tr.ravel(),n_tr.ravel()).reshape(9,9))
    deps_p_deps += sqrt32 * np.kron((2.*mu * sqrt32/ c0) * n_tr.ravel(), n_tr.ravel()).reshape(9,9)

    deqps_deqps_n = 3.*mu / c0
    deqps_deps_p_n = -(2.*mu * sqrt32/ c0)*n_tr
    deqps_dc_n = np.append(deqps_deqps_n, deqps_deps_p_n.ravel())

    deps_p_deqps_n = (3.*mu / c0 - 1.) * sqrt32 * n_tr
    deps_p_deps_p_n = np.eye(9,9) - (sqrt32 * deqps / np.sqrt(dev_eps_e_tr_squared)) * (dev_projection - np.kron(n_tr.ravel(),n_tr.ravel()).reshape(9,9))
    deps_p_deps_p_n -= sqrt32 * np.kron((2.*mu * sqrt32/ c0) * n_tr.ravel(), n_tr.ravel()).reshape(9,9)
    deps_p_dc_n = np.hstack((deps_p_deqps_n.ravel()[:,None],deps_p_deps_p_n))

    return np.vstack((deqps_deps.ravel(), deps_p_deps)), np.vstack((deqps_dc_n, deps_p_dc_n))


class J2MaterialPointUpdateGradsFixture(TestFixture):
    def setUp(self):
        E = 100.0
        poisson = 0.321
        Y0 = 0.2*E
        H = 1.0e-2*E

        self.props = {'elastic modulus': E,
                      'poisson ratio': poisson,
                      'yield strength': Y0,
                      'kinematics': 'small deformations',
                      'hardening model': 'linear',
                      'hardening modulus': H}

        materialModel = J2.create_material_model_functions(self.props)

        self.compute_state_new = jax.jit(materialModel.compute_state_new)
        self.compute_initial_state = materialModel.compute_initial_state

        # self.compute_state_new_derivs = jax.jit(jax.jacrev(self.compute_state_new, (0, 1)))
        self.compute_state_new_derivs = jax.jit(jax.jacfwd(self.compute_state_new, (0, 1)))

    def test_jax_computation_of_state_derivs_at_elastic_step(self):
        dt = 1.0

        initial_state = self.compute_initial_state()
        dc_dugrad, dc_dc_n = self.compute_state_new_derivs(np.zeros((3,3)), initial_state, dt)

        # Check derivative sizes
        self.assertEqual(dc_dugrad.shape, (10,3,3))
        self.assertEqual(dc_dc_n.shape, (10,10))

        # check derivatives w.r.t. displacement grad
        self.assertArrayNear(dc_dugrad.ravel(), np.zeros((10,3,3)).ravel(), 12)

        # check derivatives w.r.t. previous step internal variables
        self.assertArrayNear(dc_dc_n.ravel(), np.eye(10).ravel(), 12)

    def test_jax_computation_of_state_derivs_at_plastic_step(self):
        dt = 1.0
        # get random displacement gradient
        key = jax.random.PRNGKey(0)
        dispGrad = jax.random.uniform(key, (3, 3))
        initial_state = self.compute_initial_state()

        state = self.compute_state_new(dispGrad, initial_state, dt)
        dc_dugrad, dc_dc_n = self.compute_state_new_derivs(dispGrad, initial_state, dt)

        # Compute data for gold values (assuming small strain and linear kinematics)
        dc_dugrad_gold, dc_dc_n_gold = small_strain_linear_hardening_analytic_gradients(dispGrad, state, initial_state, self.props)

        # Check derivative sizes
        self.assertEqual(dc_dugrad.shape, (10,3,3))
        self.assertEqual(dc_dc_n.shape, (10,10))

        # check derivatives w.r.t. displacement grad
        self.assertArrayNear(dc_dugrad[0,:,:].ravel(), dc_dugrad_gold[0,:].ravel(), 12)
        self.assertArrayNear(dc_dugrad[1:,:,:].ravel(), dc_dugrad_gold[1:,:].ravel() , 12)

        # check derivatives w.r.t. previous step internal variables
        self.assertNear(dc_dc_n[0,0], dc_dc_n_gold[0,0], 12)
        self.assertArrayNear(dc_dc_n[0,1:], dc_dc_n_gold[0,1:], 12)
        self.assertArrayNear(dc_dc_n[1:,0], dc_dc_n_gold[1:,0], 12)
        self.assertArrayNear(dc_dc_n[1:,1:].ravel(), dc_dc_n_gold[1:,1:].ravel() , 12)

class J2GlobalMeshUpdateGradsFixture(MeshFixture):
    def setUp(self):
        self.dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        self.mesh, self.U = self.create_mesh_and_disp(4,4,[0.,1.],[0.,1.],
                                            lambda x: self.dispGrad0@x)

        E = 100.0
        poisson = 0.321
        H = 1e-2 * E
        Y0 = 0.3 * E

        self.props = {'elastic modulus': E,
                 'poisson ratio': poisson,
                 'yield strength': Y0,
                 'kinematics': 'small deformations',
                 'hardening model': 'linear',
                 'hardening modulus': H}

        self.materialModel = J2.create_material_model_functions(self.props)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)

        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        
        self.mechFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                         "plane strain",
                                                         self.materialModel)
                
        EBCs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]

        self.dofManager = FunctionSpace.DofManager(self.fs, 2, EBCs)

        self.Ubc = self.dofManager.get_bc_values(self.U)

    def test_state_derivs_at_elastic_step(self):

        internalVariables = self.mechFuncs.compute_initial_state()

        p = Objective.Params(None, internalVariables, None, None, None)

        def update_internal_vars_test(Uu, internalVars):
            U = self.dofManager.create_field(Uu)
            internalVariablesNew = self.mechFuncs.compute_updated_internal_variables(U, internalVars)
            return internalVariablesNew

        Uu = 0.0*self.dofManager.get_unknown_values(self.U) 

        update_internal_variables_derivs = jax.jacfwd(update_internal_vars_test, (0,1))
        dc_du, dc_dc_n = update_internal_variables_derivs(Uu, p[1])

        nElems = Mesh.num_elements(self.mesh)
        nQpsPerElem = QuadratureRule.len(self.quadRule)
        nIntVars = 10
        nFreeDofs = Uu.shape[0]

        self.assertEqual(dc_du.shape, (nElems,nQpsPerElem,nIntVars,nFreeDofs))
        self.assertEqual(dc_dc_n.shape, (nElems,nQpsPerElem,nIntVars,nElems,nQpsPerElem,nIntVars))

        for i in range(0,nElems):
            for j in range(0,nQpsPerElem):
                self.assertArrayNear(dc_du[i,j,:,:].ravel(), np.zeros((nIntVars,nFreeDofs)).ravel(), 12)
                self.assertArrayNear(dc_dc_n[i,j,:,i,j,:].ravel(), np.eye(nIntVars).ravel(), 12)

    def test_state_derivs_at_plastic_step(self):

        initialInternalVariables = self.mechFuncs.compute_initial_state()

        p = Objective.Params(None, initialInternalVariables, None, None, None)

        def update_internal_vars_test(U, internalVars):
            internalVariablesNew = self.mechFuncs.compute_updated_internal_variables(U, internalVars)
            return internalVariablesNew

        def compute_energy(Uu, p):
            U = self.dofManager.create_field(Uu, self.Ubc)
            internalVariables = p[1]
            return self.mechFuncs.compute_strain_energy(U, internalVariables)

        UuGuess = 0.0*self.dofManager.get_unknown_values(self.U) 
        objective = Objective.Objective(compute_energy, UuGuess, p)
        Uu = EqSolver.nonlinear_equation_solve(objective, UuGuess, p, EqSolver.get_settings(), useWarmStart=False)
        U = self.dofManager.create_field(Uu, self.Ubc)

        internalVariables = update_internal_vars_test(U, p[1])

        update_internal_variables_derivs = jax.jacfwd(update_internal_vars_test, (0,1))
        dc_du, dc_dc_n = update_internal_variables_derivs(U, p[1])

        nElems = Mesh.num_elements(self.mesh)
        nQpsPerElem = QuadratureRule.len(self.quadRule)
        nIntVars = 10

        self.assertEqual(dc_du.shape, (nElems,nQpsPerElem,nIntVars,U.shape[0],U.shape[1]))
        self.assertEqual(dc_dc_n.shape, (nElems,nQpsPerElem,nIntVars,nElems,nQpsPerElem,nIntVars))

        for i in range(0,nElems):
            for j in range(0,nQpsPerElem):
                state = internalVariables[i,j,:]
                initial_state = initialInternalVariables[i,j,:]

                conn = self.mesh.conns[i]
                Uele = U[conn]
                shapeGrads = self.fs.shapeGrads[i,j,:,:]
                dispGrad = TensorMath.tensor_2D_to_3D(np.tensordot(Uele,shapeGrads,axes=[0,0]))
                Be_mat = np.zeros((9,6)).\
                         at[(0,3),(0,1)].set(shapeGrads[0,0]).at[(1,4),(0,1)].set(shapeGrads[0,1]).\
                         at[(0,3),(2,3)].set(shapeGrads[1,0]).at[(1,4),(2,3)].set(shapeGrads[1,1]).\
                         at[(0,3),(4,5)].set(shapeGrads[2,0]).at[(1,4),(4,5)].set(shapeGrads[2,1])
                        
                dc_dugrad_gold, dc_dc_n_gold = small_strain_linear_hardening_analytic_gradients(dispGrad, state, initial_state, self.props)

                self.assertArrayNear(dc_dc_n[i,j,:,i,j,:].ravel(), dc_dc_n_gold.ravel(), 10)

                dc_duele_gold = np.tensordot(dc_dugrad_gold, Be_mat, axes=1)
                dc_er_du = dc_du[i,j,:,:,:]
                self.assertArrayNear(dc_er_du[:,conn,:].ravel(), dc_duele_gold.reshape(10,3,2).ravel(), 10)


class J2GlobalMeshAdjointSolveFixture(MeshFixture):
    def setUp(self):
        self.dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        self.mesh, self.U = self.create_mesh_and_disp(4,4,[0.,1.],[0.,1.],
                                            lambda x: self.dispGrad0@x)

        E = 100.0
        poisson = 0.321
        H = 1e-2 * E
        Y0 = 0.3 * E

        self.props = {'elastic modulus': E,
                 'poisson ratio': poisson,
                 'yield strength': Y0,
                 'kinematics': 'small deformations',
                 'hardening model': 'linear',
                 'hardening modulus': H}

        self.materialModel = J2.create_material_model_functions(self.props)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)

        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        
        self.mechFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                         "plane strain",
                                                         self.materialModel)
                
        EBCs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]

        self.dofManager = FunctionSpace.DofManager(self.fs, 2, EBCs)

        self.Ubc = self.dofManager.get_bc_values(self.U)
    
    def test_adjoint_solve(self):

        # solve forward problem
        steps = 2
        Ubc_inc = self.Ubc / steps
        ivs = self.mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=np.zeros(self.Ubc.shape), state_data=ivs)

        def compute_energy(Uu, p):
            U = self.dofManager.create_field(Uu, p.bc_data)
            internalVariables = p.state_data
            return self.mechFuncs.compute_strain_energy(U, internalVariables)

        Uu = 0.0*self.dofManager.get_unknown_values(self.U) 
        objective = Objective.Objective(compute_energy, Uu, p)

        storedState = []
        storedState.append((Uu, p))

        for step in range(1, steps+1):
            p = Objective.param_index_update(p, 0, step*Ubc_inc)
            Uu = EqSolver.nonlinear_equation_solve(objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = self.dofManager.create_field(Uu, p.bc_data)
            ivs = self.mechFuncs.compute_updated_internal_variables(U, p.state_data)
            p = Objective.param_index_update(p, 1, ivs)
            storedState.append((Uu, p))

        # define functions for gradients
        internal_vars_shape = ivs.shape
        coords_shape = self.mesh.coords.shape

        def update_internal_vars_test(Uu, ivs, coords):
            U = self.dofManager.create_field(Uu, p.bc_data)
            internal_vars = ivs.reshape(internal_vars_shape)
            coordinates = coords.reshape(coords_shape)
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coordinates, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
            return mech_funcs.compute_updated_internal_variables(U, internal_vars).ravel()

        compute_dc = jax.jacfwd(update_internal_vars_test, (0, 1, 2))

        # for now just using sum of incremental internal energy as objective
        def energy_function_coords(Uu, ivs, ivs_old, p, coords):
            U = self.dofManager.create_field(Uu, p.bc_data)
            internal_vars = ivs.reshape(internal_vars_shape)
            coordinates = coords.reshape(coords_shape)
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coordinates, self.mesh, self.quadRule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.materialModel)
            return mech_funcs.compute_strain_energy(U, internal_vars)

        compute_df = jax.grad(energy_function_coords, (0, 1, 2, 4))

        # initialize
        gradient = np.zeros(np.prod(np.array(coords_shape)))
        mu = np.zeros(np.prod(np.array(internal_vars_shape))) 
        adjointLoad = np.zeros(Uu.shape)

        coords = self.mesh.coords.ravel()
        for step in reversed(range(1, steps+1)):
            Uu = storedState[step][0]
            p = storedState[step][1]
            ivs = p.state_data.ravel()
            ivs_old = storedState[step-1][1].state_data.ravel()

            # partial derivatives of internal variables update
            dc_du, dc_dcn, dc_dx = compute_dc(Uu, ivs_old, coords)

            # partial derivatives of objective increment
            df_du, df_dc, df_dcn, df_dx = compute_df(Uu, ivs, ivs_old, p, coords)

            # Compute adjoint load
            mu += df_dc
            adjointLoad -= df_du
            adjointLoad -= np.tensordot(mu, dc_du, axes=1) # mu^T dc/du

            # Solve adjoint equation
            n = self.dofManager.get_unknown_size()
            objective.p = p # have to update parameters to get precond to work
            objective.update_precond(Uu) # update preconditioner for use in cg (will converge in 1 iteration as long as the preconditioner is not approximate)
            dRdu = linalg.LinearOperator((n, n), lambda V: onp.asarray(objective.hessian_vec(Uu, V)))
            dRdu_decomp = linalg.LinearOperator((n, n), lambda V: onp.asarray(objective.apply_precond(V)))
            adjointVector = linalg.cg(dRdu, onp.array(adjointLoad, copy=False), tol=1e-10, atol=0.0, M=dRdu_decomp)[0]

            # Update gradient
            gradient += df_dx
            # The action dRdX * lambda (the same as lambda^T * dRdX)
            gradient += jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(Uu, ivs, ivs_old, p, z), coords)[1](adjointVector)[0]
            gradient += np.tensordot(mu, dc_dx, axes=1) # mu^T dc/dx

            # Update mu
            mu = np.tensordot(mu, dc_dcn, axes=1)
            # The action dRdcn * lambda (the same as lambda^T * dRdcn)
            mu += jax.vjp(lambda z: jax.grad(energy_function_coords, 0)(Uu, ivs, z, p, coords), ivs_old)[1](adjointVector)[0]
            mu += df_dcn

        
if __name__ == '__main__':
    unittest.main()
