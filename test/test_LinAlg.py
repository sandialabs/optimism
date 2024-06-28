import jax
from jax import numpy as np
from jax.test_util import check_grads
from scipy.spatial.transform import Rotation
import unittest

from optimism import LinAlg
from .TestFixture import TestFixture

def generate_n_random_symmetric_matrices(n, minval=0.0, maxval=1.0):
    key = jax.random.PRNGKey(0)
    As = jax.random.uniform(key, (n,3,3), minval=minval, maxval=maxval)
    return jax.vmap(lambda A: np.dot(A.T,A), (0,))(As)

sqrtm_jit = jax.jit(LinAlg.sqrtm)
logm_iss_jit = jax.jit(LinAlg.logm_iss)

class TestLinAlg(TestFixture):
    def setUp(self):
        self.sym_mat = generate_n_random_symmetric_matrices(1)[0]
        # make a matrix with 2 identical eigenvalues
        R = Rotation.random(random_state=41).as_matrix()
        eigvals = np.array([2., 0.5, 2.])
        self.sym_mat_double_degeneracy = R@np.diag(eigvals)@R.T
 
    ### sqrtm ###
    
    def test_sqrtm_jit(self):
        sqrtC = sqrtm_jit(self.sym_mat)
        self.assertTrue(not np.isnan(sqrtC).any())


    def test_sqrtm(self):
        mats = generate_n_random_symmetric_matrices(100)
        sqrtMats = jax.vmap(sqrtm_jit, (0,))(mats)
        shouldBeMats = jax.vmap(lambda A: np.dot(A,A), (0,))(sqrtMats)
        self.assertArrayNear(shouldBeMats, mats, 10)


    def test_sqrtm_fwd_mode_derivative(self):
        check_grads(LinAlg.sqrtm, (self.sym_mat,), order=2, modes=["fwd"])


    def test_sqrtm_rev_mode_derivative(self):
        check_grads(LinAlg.sqrtm, (self.sym_mat,), order=2, modes=["rev"])


    def test_sqrtm_on_degenerate_eigenvalues(self):
        C = self.sym_mat_double_degeneracy
        sqrtC = LinAlg.sqrtm(C)
        shouldBeC = np.dot(sqrtC, sqrtC)
        self.assertArrayNear(shouldBeC, C, 12)
        check_grads(LinAlg.sqrtm, (C,), order=2, modes=["rev"])


    def test_sqrtm_on_10x10(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
        C = F.T@F
        sqrtC = LinAlg.sqrtm(C)
        shouldBeC = np.dot(sqrtC,sqrtC)
        self.assertArrayNear(shouldBeC, C, 11)


    def test_sqrtm_derivatives_on_10x10(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
        C = F.T@F
        check_grads(LinAlg.sqrtm, (C,), order=1, modes=["fwd", "rev"])

    ### sqrtm ###

    def test_logm_iss_on_matrix_near_identity(self):
        key = jax.random.PRNGKey(0)
        id_perturbation = 1.0 + jax.random.uniform(key, (3,), minval=1e-8, maxval=0.01)
        A = np.diag(id_perturbation)
        logA = LinAlg.logm_iss(A)
        self.assertArrayNear(logA, np.diag(np.log(id_perturbation)), 12)


    def test_logm_iss_on_double_degenerate_eigenvalues(self):
        C = self.sym_mat_double_degeneracy
        logC = LinAlg.logm_iss(C)
        explogC = jax.scipy.linalg.expm(logC)
        self.assertArrayNear(C, explogC, 8)


    def test_logm_iss_on_triple_degenerate_eigvalues(self):
        A = 4.0*np.identity(3)
        logA = LinAlg.logm_iss(A)
        self.assertArrayNear(logA, np.log(4.0)*np.identity(3), 12)


    def test_logm_iss_jit(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        logC = logm_iss_jit(C)
        self.assertFalse(np.isnan(logC).any())


    def test_logm_iss_on_full_3x3s(self):
        mats = generate_n_random_symmetric_matrices(1000)
        logMats = jax.vmap(logm_iss_jit, (0,))(mats)
        shouldBeMats = jax.vmap(lambda A: jax.scipy.linalg.expm(A), (0,))(logMats)
        self.assertArrayNear(shouldBeMats, mats, 7)      

        
    def test_logm_iss_fwd_mode_derivative(self):
        check_grads(logm_iss_jit, (self.sym_mat,), order=1, modes=['fwd'])


    def test_logm_iss_rev_mode_derivative(self):
        check_grads(logm_iss_jit, (self.sym_mat,), order=1, modes=['rev'])


    def test_logm_iss_hessian_on_double_degenerate_eigenvalues(self):
        C = self.sym_mat_double_degeneracy
        check_grads(jax.jacrev(LinAlg.logm_iss), (C,), order=1, modes=['fwd'], rtol=1e-9, atol=1e-9, eps=1e-5)


    def test_logm_iss_derivatives_on_double_degenerate_eigenvalues(self):
        C = self.sym_mat_double_degeneracy
        check_grads(LinAlg.logm_iss, (C,), order=1, modes=['fwd'])
        check_grads(LinAlg.logm_iss, (C,), order=1, modes=['rev'])


    def test_logm_iss_derivatives_on_triple_degenerate_eigenvalues(self):
        A = 4.0*np.identity(3)
        check_grads(LinAlg.logm_iss, (A,), order=1, modes=['fwd'])
        check_grads(LinAlg.logm_iss, (A,), order=1, modes=['rev'])


    def test_logm_iss_on_10x10(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
        C = F.T@F
        logC = LinAlg.logm_iss(C)
        explogC = jax.scipy.linalg.expm(logC)
        self.assertArrayNear(explogC, C, 8)