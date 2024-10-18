import unittest

import jax
import jax.numpy as np

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism.material import J2Plastic as J2
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import TensorMath

from optimism.test.TestFixture import TestFixture
from optimism.test.MeshFixture import MeshFixture

from optimism.inverse import MechanicsInverse

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
    @classmethod
    def setUpClass(cls):
        dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        mf = MeshFixture()
        cls.mesh, cls.U = mf.create_mesh_and_disp(4,4,[0.,1.],[0.,1.],
                                            lambda x: dispGrad0@x)

        E = 100.0
        poisson = 0.321
        H = 1e-2 * E
        Y0 = 0.3 * E

        cls.props = {'elastic modulus': E,
                 'poisson ratio': poisson,
                 'yield strength': Y0,
                 'kinematics': 'small deformations',
                 'hardening model': 'linear',
                 'hardening modulus': H}

        cls.materialModel = J2.create_material_model_functions(cls.props)
        cls.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        cls.fs = FunctionSpace.construct_function_space(cls.mesh, cls.quadRule)
        cls.mechFuncs = Mechanics.create_mechanics_functions(cls.fs,
                                                         "plane strain",
                                                         cls.materialModel)
        cls.ivs_prev = cls.mechFuncs.compute_initial_state()
                
        EBCs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        cls.dofManager = FunctionSpace.DofManager(cls.fs, 2, EBCs)
        cls.Ubc = cls.dofManager.get_bc_values(cls.U)

        p = Objective.Params(None, cls.ivs_prev, None, None, None)
        UuGuess = 0.0*cls.dofManager.get_unknown_values(cls.U) 

        def compute_energy(Uu, p):
            U = cls.dofManager.create_field(Uu, cls.Ubc)
            internalVariables = p[1]
            return cls.mechFuncs.compute_strain_energy(U, internalVariables)

        objective = Objective.Objective(compute_energy, UuGuess, p)
        cls.Uu, solverSuccess = EqSolver.nonlinear_equation_solve(objective, UuGuess, p, EqSolver.get_settings(), useWarmStart=False)
        U = cls.dofManager.create_field(cls.Uu, cls.Ubc)
        cls.ivs = cls.mechFuncs.compute_updated_internal_variables(U, cls.ivs_prev)

    def test_state_derivs_at_elastic_step(self):

        def update_internal_vars_test(Uu, ivs_prev):
            U = self.dofManager.create_field(Uu)
            ivs = self.mechFuncs.compute_updated_internal_variables(U, ivs_prev)
            return ivs

        Uu = 0.0*self.dofManager.get_unknown_values(self.U) 

        update_internal_variables_derivs = jax.jacfwd(update_internal_vars_test, (0,1))
        dc_du, dc_dc_n = update_internal_variables_derivs(Uu, self.ivs_prev)

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

        def update_internal_vars_test(U, ivs_prev):
            ivs = self.mechFuncs.compute_updated_internal_variables(U, ivs_prev)
            return ivs

        update_internal_variables_derivs = jax.jacfwd(update_internal_vars_test, (0,1))

        U = self.dofManager.create_field(self.Uu, self.Ubc)
        dc_du, dc_dc_n = update_internal_variables_derivs(U, self.ivs_prev)

        nElems = Mesh.num_elements(self.mesh)
        nQpsPerElem = QuadratureRule.len(self.quadRule)
        nIntVars = 10
        nDims = 2
        nNodes = Mesh.num_nodes(self.mesh)

        self.assertEqual(dc_du.shape, (nElems,nQpsPerElem,nIntVars,nNodes,nDims))
        self.assertEqual(dc_dc_n.shape, (nElems,nQpsPerElem,nIntVars,nElems,nQpsPerElem,nIntVars))

        for i in range(0,nElems):
            for j in range(0,nQpsPerElem):
                state = self.ivs[i,j,:]
                initial_state = self.ivs_prev[i,j,:]

                conn = self.mesh.conns[i]
                Uele = U[conn]
                shapeGrads = self.fs.shapeGrads[i,j,:,:]
                dispGrad = TensorMath.tensor_2D_to_3D(np.tensordot(Uele,shapeGrads,axes=[0,0]))
                Be_mat = np.zeros((9,6)).\
                         at[(0,3),(0,1)].set(shapeGrads[0,0]).at[(1,4),(0,1)].set(shapeGrads[0,1]).\
                         at[(0,3),(2,3)].set(shapeGrads[1,0]).at[(1,4),(2,3)].set(shapeGrads[1,1]).\
                         at[(0,3),(4,5)].set(shapeGrads[2,0]).at[(1,4),(4,5)].set(shapeGrads[2,1])
                        
                dc_dugrad_gold, dc_dc_n_gold = small_strain_linear_hardening_analytic_gradients(dispGrad, state, initial_state, self.props)

                dc_duele_gold = np.tensordot(dc_dugrad_gold, Be_mat, axes=1)
                dc_er_du = dc_du[i,j,:,:,:]

                self.assertArrayNear(dc_er_du[:,conn,:].ravel(), dc_duele_gold.reshape(10,3,2).ravel(), 10)
                self.assertArrayNear(np.delete(dc_er_du, conn, axis=1).ravel(), np.zeros((10,nNodes-conn.shape[0],2)).ravel(), 10)

                for p in range(0,nElems):
                    for q in range(0,nQpsPerElem):
                        if(i == p and j == q):
                            self.assertArrayNear(dc_dc_n[i,j,:,i,j,:].ravel(), dc_dc_n_gold.ravel(), 10)
                        else:
                            self.assertArrayNear(dc_dc_n[i,j,:,p,q,:].ravel(), np.zeros((nIntVars,nIntVars)).ravel(), 10)

    def test_state_derivs_computed_locally_at_plastic_step(self):

        ivsUpdateInverseFuncs = MechanicsInverse.create_ivs_update_inverse_functions(self.fs,
                                                                                     "plane strain",
                                                                                     self.materialModel)

        U = self.dofManager.create_field(self.Uu, self.Ubc)
        dc_dc_n = ivsUpdateInverseFuncs.ivs_update_jac_ivs_prev(U, self.ivs_prev)

        nElems = Mesh.num_elements(self.mesh)
        nQpsPerElem = QuadratureRule.len(self.quadRule)
        nIntVars = 10

        self.assertEqual(dc_dc_n.shape, (nElems,nQpsPerElem,nIntVars,nIntVars))

        for i in range(0,nElems):
            for j in range(0,nQpsPerElem):
                state = self.ivs[i,j,:]
                initial_state = self.ivs_prev[i,j,:]

                conn = self.mesh.conns[i]
                Uele = U[conn]
                shapeGrads = self.fs.shapeGrads[i,j,:,:]
                dispGrad = TensorMath.tensor_2D_to_3D(np.tensordot(Uele,shapeGrads,axes=[0,0]))

                _, dc_dc_n_gold = small_strain_linear_hardening_analytic_gradients(dispGrad, state, initial_state, self.props)

                self.assertArrayNear(dc_dc_n[i,j,:,:].ravel(), dc_dc_n_gold.ravel(), 10)


        
if __name__ == '__main__':
    unittest.main()
