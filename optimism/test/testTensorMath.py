import unittest
from scipy.spatial.transform import Rotation

import jax
from jax import numpy as np
from jax.test_util import check_grads
from jax.scipy import linalg

from optimism.test.TestFixture import TestFixture
from optimism import TensorMath

R = Rotation.random(random_state=41).as_matrix()

def numerical_grad(f):
    def lam(A):
        df = np.zeros((3,3))
        eps = 1e-7
        ff = f(A)
        for i in range(3):
            for j in range(3):
                Ap = A.at[i,j].add(eps)
                fp = f(Ap)
                fprime = (fp-ff)/eps
                df = df.at[i,j].add(fprime)
        return df
    return lam


def generate_n_random_symmetric_matrices(n, minval=0.0, maxval=1.0):
    key = jax.random.PRNGKey(0)
    As = jax.random.uniform(key, (n,3,3), minval=minval, maxval=maxval)
    return jax.vmap(lambda A: np.dot(A.T,A), (0,))(As)


class TensorMathFixture(TestFixture):

    def setUp(self):
        self.log_squared = lambda A: np.tensordot(TensorMath.log_sqrt(A), TensorMath.log_sqrt(A))
        self.sqrtm_jit = jax.jit(TensorMath.sqrtm)
        self.logm_iss_jit = jax.jit(TensorMath.logm_iss)
        

    def test_log_sqrt_tensor_jvp_0(self):
        A = np.array([ [2.0, 0.0, 0.0],
                       [0.0, 1.2, 0.0],
                       [0.0, 0.0, 2.0] ])

        check_grads(self.log_squared, (A,), order=1)
        
        
    def test_log_sqrt_tensor_jvp_1(self):
        A = np.array([ [2.0, 0.0, 0.0],
                       [0.0, 1.2, 0.0],
                       [0.0, 0.0, 3.0] ])

        check_grads(self.log_squared, (A,), order=1)

        
    def test_log_sqrt_tensor_jvp_2(self):
        A = np.array([ [2.0, 0.0, 0.2],
                       [0.0, 1.2, 0.1],
                       [0.2, 0.1, 3.0] ])

        check_grads(self.log_squared, (A,), order=1)


    @unittest.expectedFailure
    def test_log_sqrt_hessian_on_double_degenerate_eigenvalues(self):
        eigvals = np.array([2., 0.5, 2.])
        C = R@np.diag(eigvals)@R.T
        check_grads(jax.jacrev(TensorMath.log_sqrt), (C,), order=1, modes=['fwd'], rtol=1e-9, atol=1e-9, eps=1e-5)


    def test_eigen_sym33_non_unit(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F
        d,vecs = TensorMath.eigen_sym33_unit(C)
        self.assertArrayNear(C, vecs@np.diag(d)@vecs.T, 13)
        self.assertArrayNear(vecs@vecs.T, np.identity(3), 13)

        
    def test_eigen_sym33_non_unit_degenerate_case(self):
        C = 5.0*np.identity(3)
        d,vecs = TensorMath.eigen_sym33_unit(C)
        self.assertArrayNear(C, vecs@np.diag(d)@vecs.T, 13)
        self.assertArrayNear(vecs@vecs.T, np.identity(3), 13)

        
    ### mtk_log_sqrt tests ###
        
        
    def test_log_sqrt_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))

        logSqrtVal = np.log(np.sqrt(val))
        self.assertArrayNear(TensorMath.mtk_log_sqrt(C), np.diag(np.array([logSqrtVal, logSqrtVal, logSqrtVal])), 12)


    def test_log_sqrt_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = R@np.diag(np.array([val1, val2, val1]))@R.T

        logSqrt1 = np.log(np.sqrt(val1))
        logSqrt2 = np.log(np.sqrt(val2))
        diagLogSqrt = np.diag(np.array([logSqrt1, logSqrt2, logSqrt1]))

        logSqrtCExpected = R@diagLogSqrt@R.T
        self.assertArrayNear(TensorMath.mtk_log_sqrt(C), logSqrtCExpected, 12)

        
    def test_log_sqrt_squared_grad_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))

        def log_squared(A):
            lg = TensorMath.mtk_log_sqrt(A)
            return np.tensordot(lg, lg)
        check_grads(log_squared, (C,), order=1)
        
        
    def test_log_sqrt_squared_grad_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = R@np.diag(np.array([val1, val2, val1]))@R.T

        def log_squared(A):
            lg = TensorMath.mtk_log_sqrt(A)
            return np.tensordot(lg, lg)
        check_grads(log_squared, (C,), order=1)

        
    def test_log_sqrt_squared_grad_rand(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F

        def log_squared(A):
            lg = TensorMath.mtk_log_sqrt(A)
            return np.tensordot(lg, lg)
        check_grads(log_squared, (C,), order=1)
        
        
    ### mtk_pow tests ###

    
    def test_pow_scaled_identity(self):
        m = 0.25
        val = 1.2
        C = np.diag(np.array([val, val, val]))

        powVal = np.power(val, m)
        self.assertArrayNear(TensorMath.mtk_pow(C,m), np.diag(np.array([powVal, powVal, powVal])), 12)


    def test_pow_double_eigs(self):
        m = 0.25
        val1 = 2.1
        val2 = 0.6
        C = R@np.diag(np.array([val1, val2, val1]))@R.T

        powVal1 = np.power(val1, m)
        powVal2 = np.power(val2, m)
        diagLogSqrt = np.diag(np.array([powVal1, powVal2, powVal1]))

        logSqrtCExpected = R@diagLogSqrt@R.T

        self.assertArrayNear(TensorMath.mtk_pow(C,m), logSqrtCExpected, 12)

        
    def test_pow_squared_grad_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))

        def pow_squared(A):
            m = 0.25
            lg = TensorMath.mtk_pow(A, m)
            return np.tensordot(lg, lg)
        check_grads(pow_squared, (C,), order=1)


    def test_pow_squared_grad_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = R@np.diag(np.array([val1, val2, val1]))@R.T

        def pow_squared(A):
            m=0.25
            lg = TensorMath.mtk_pow(A, m)
            return np.tensordot(lg, lg)
        check_grads(pow_squared, (C,), order=1)


    def test_pow_squared_grad_rand(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F

        def pow_squared(A):
            m=0.25
            lg = TensorMath.mtk_pow(A, m)
            return np.tensordot(lg, lg)
        check_grads(pow_squared, (C,), order=1)
        
        
    ### sqrtm ###

    
    def test_sqrtm_jit(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        sqrtC = self.sqrtm_jit(C)
        self.assertFalse(np.isnan(sqrtC).any())
        

    def test_sqrtm(self):
        mats = generate_n_random_symmetric_matrices(100)
        sqrtMats = jax.vmap(self.sqrtm_jit, (0,))(mats)
        shouldBeMats = jax.vmap(lambda A: np.dot(A,A), (0,))(sqrtMats)
        self.assertArrayNear(shouldBeMats, mats, 10)


    def test_sqrtm_fwd_mode_derivative(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        check_grads(TensorMath.sqrtm, (C,), order=2, modes=["fwd"])


    def test_sqrtm_rev_mode_derivative(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        check_grads(TensorMath.sqrtm, (C,), order=2, modes=["rev"])


    def test_sqrtm_on_degenerate_eigenvalues(self):
        C = R@np.diag(np.array([2., 0.5, 2]))@R.T
        sqrtC = TensorMath.sqrtm(C)
        shouldBeC = np.dot(sqrtC, sqrtC)
        self.assertArrayNear(shouldBeC, C, 12)
        check_grads(TensorMath.sqrtm, (C,), order=2, modes=["rev"])


    def test_sqrtm_on_10x10(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
        C = F.T@F
        sqrtC = TensorMath.sqrtm(C)
        shouldBeC = np.dot(sqrtC,sqrtC)
        self.assertArrayNear(shouldBeC, C, 12)


    def test_sqrtm_derivatives_on_10x10(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
        C = F.T@F
        check_grads(TensorMath.sqrtm, (C,), order=1, modes=["fwd", "rev"])


    def test_logm_iss_on_matrix_near_identity(self):
        key = jax.random.PRNGKey(0)
        id_perturbation = 1.0 + jax.random.uniform(key, (3,), minval=1e-8, maxval=0.01)
        A = np.diag(id_perturbation)
        logA = TensorMath.logm_iss(A)
        self.assertArrayNear(logA, np.diag(np.log(id_perturbation)), 12)


    def test_logm_iss_on_double_degenerate_eigenvalues(self):
        eigvals = np.array([2., 0.5, 2.])
        C = R@np.diag(eigvals)@R.T
        logC = TensorMath.logm_iss(C)
        logCSpectral = R@np.diag(np.log(eigvals))@R.T
        self.assertArrayNear(logC, logCSpectral, 12)


    def test_logm_iss_on_triple_degenerate_eigvalues(self):
        A = 4.0*np.identity(3)
        logA = TensorMath.logm_iss(A)
        self.assertArrayNear(logA, np.log(4.0)*np.identity(3), 12)


    def test_logm_iss_jit(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        logC = self.logm_iss_jit(C)
        self.assertFalse(np.isnan(logC).any())


    def test_logm_iss_on_full_3x3s(self):
        mats = generate_n_random_symmetric_matrices(1000)
        logMats = jax.vmap(self.logm_iss_jit, (0,))(mats)
        shouldBeMats = jax.vmap(lambda A: linalg.expm(A), (0,))(logMats)
        self.assertArrayNear(shouldBeMats, mats, 7)      

        
    def test_logm_iss_fwd_mode_derivative(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        check_grads(self.logm_iss_jit, (C,), order=1, modes=['fwd'])


    def test_logm_iss_rev_mode_derivative(self):
        C = generate_n_random_symmetric_matrices(1)[0]
        check_grads(self.logm_iss_jit, (C,), order=1, modes=['rev'])


    def test_logm_iss_hessian_on_double_degenerate_eigenvalues(self):
        eigvals = np.array([2., 0.5, 2.])
        C = R@np.diag(eigvals)@R.T
        check_grads(jax.jacrev(TensorMath.logm_iss), (C,), order=1, modes=['fwd'], rtol=1e-9, atol=1e-9, eps=1e-5)


    def test_logm_iss_derivatives_on_double_degenerate_eigenvalues(self):
        eigvals = np.array([2., 0.5, 2.])
        C = R@np.diag(eigvals)@R.T
        check_grads(TensorMath.logm_iss, (C,), order=1, modes=['fwd'])
        check_grads(TensorMath.logm_iss, (C,), order=1, modes=['rev'])


    def test_logm_iss_derivatives_on_triple_degenerate_eigenvalues(self):
        A = 4.0*np.identity(3)
        check_grads(TensorMath.logm_iss, (A,), order=1, modes=['fwd'])
        check_grads(TensorMath.logm_iss, (A,), order=1, modes=['rev'])


    def test_logm_iss_on_10x10(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (10,10), minval=1e-8, maxval=10.0)
        C = F.T@F
        logC = TensorMath.logm_iss(C)
        logCSpectral = TensorMath.logh(C)
        self.assertArrayNear(logC, logCSpectral, 12)

        
if __name__ == '__main__':
    unittest.main()
