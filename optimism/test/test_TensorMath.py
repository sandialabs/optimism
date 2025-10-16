import unittest
from scipy.spatial.transform import Rotation

import jax
from jax import numpy as np
from jax.test_util import check_grads

from .TestFixture import TestFixture
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


class TensorMathFixture(TestFixture):

    def setUp(self):
        key = jax.random.PRNGKey(0)
        self.R = jax.random.orthogonal(key, 3)
        self.assertGreater(np.linalg.det(self.R), 0) # make sure this is a rotation and not a reflection
        self.log_squared = lambda A: np.tensordot(TensorMath.log_sqrt(A), TensorMath.log_sqrt(A))
        

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

    # log_symm tests

    def test_log_symm_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        logVal = np.log(val)
        self.assertArrayNear(TensorMath.log_symm(C), np.diag(np.array([logVal, logVal, logVal])), 12)

    def test_log_symm_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T

        log1 = np.log(val1)
        log2 = np.log(val2)
        diagLog = np.diag(np.array([log1, log2, log1]))

        logCExpected = self.R@diagLog@self.R.T
        self.assertArrayNear(TensorMath.log_symm(C), logCExpected, 12)

    def test_log_symm_gradient_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        check_grads(TensorMath.log_symm, (C,), order=1)

    def test_log_symm_gradient_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        check_grads(TensorMath.log_symm, (C,), order=1)

    def test_log_symm_gradient_distinct_eigenvalues(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F
        check_grads(TensorMath.log_symm, (C,), order=1)

    def test_log_symm_gradient_almost_double_degenerate(self):
        C = self.R@np.diag(np.array([2.1, 2.1 + 1e-8, 3.0]))@self.R.T
        check_grads(TensorMath.log_symm, (C,), order=1, atol=1e-16, eps=1e-10)

    # sqrt_symm_tests

    def test_sqrt_symm(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F
        U = TensorMath.sqrt_symm(C)
        self.assertArrayNear(U@U, C, 12)

    def test_sqrt_symm_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        sqrtVal = np.sqrt(val)
        self.assertArrayNear(TensorMath.sqrt_symm(C), np.diag(np.array([sqrtVal, sqrtVal, sqrtVal])), 12)

    def test_sqrt_symm_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        sqrt1 = np.sqrt(val1)
        sqrt = np.sqrt(val2)
        diagSqrt = np.diag(np.array([sqrt1, sqrt, sqrt1]))

        sqrtCExpected = self.R@diagSqrt@self.R.T
        self.assertArrayNear(TensorMath.sqrt_symm(C), sqrtCExpected, 12)

    def test_sqrt_symm_gradient_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        check_grads(TensorMath.sqrt_symm, (C,), order=1)

    def test_sqrt_symm_gradient_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        check_grads(TensorMath.sqrt_symm, (C,), order=1)

    def test_sqrt_symm_gradient_distinct_eigenvalues(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F
        check_grads(TensorMath.sqrt_symm, (C,), order=1)

    def test_sqrt_symm_gradient_almost_double_degenerate(self):
        C = self.R@np.diag(np.array([2.1, 2.1 + 1e-8, 3.0]))@self.R.T
        check_grads(TensorMath.sqrt_symm, (C,), order=1, eps=1e-10)

    ### exp_symm tests
    def test_exp_symm_at_identity(self):
        I = TensorMath.exp_symm(np.zeros((3, 3)))
        self.assertArrayNear(I, np.identity(3), 12)

    def test_exp_symm_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        expVal = np.exp(val)
        self.assertArrayNear(TensorMath.exp_symm(C), np.diag(np.array([expVal, expVal, expVal])), 12)

    def test_exp_symm_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        exp1 = np.exp(val1)
        exp2 = np.exp(val2)
        diagExp = np.diag(np.array([exp1, exp2, exp1]))
        expCExpected = self.R@diagExp@self.R.T
        self.assertArrayNear(TensorMath.exp_symm(C), expCExpected, 12)

    def test_exp_symm_gradient_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        check_grads(TensorMath.exp_symm, (C,), order=1)

    def test_exp_symm_gradient_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        check_grads(TensorMath.exp_symm, (C,), order=1)

    def test_exp_symm_gradient_distinct_eigenvalues(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F
        check_grads(TensorMath.exp_symm, (C,), order=1)

    def test_sqrt_symm_gradient_almost_double_degenerate(self):
        C = self.R@np.diag(np.array([2.1, 2.1 + 1e-8, 3.0]))@self.R.T
        check_grads(TensorMath.exp_symm, (C,), order=1, eps=1e-8, rtol=5e-5)
    
    # pow_symm tests

    def test_pow_symm_scaled_identity(self):
        val = 1.2
        C = val*np.identity(3)
        m = 3
        powVal = np.power(val, m)
        self.assertArrayNear(TensorMath.pow_symm(C, m), np.diag(np.array([powVal, powVal, powVal])), 12)

    def test_pow_symm_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        m = 0.25
        pow1 = np.power(val1, m)
        pow2 = np.power(val2, m)
        diagPow = np.diag(np.array([pow1, pow2, pow1]))
        powCExpected = self.R@diagPow@self.R.T
        self.assertArrayNear(TensorMath.pow_symm(C, m), powCExpected, 12)

    def test_pow_symm_gradient_scaled_identity(self):
        val = 1.2
        C = np.diag(np.array([val, val, val]))
        m = 3
        check_grads(lambda A: TensorMath.pow_symm(A, m), (C,), order=1)

    def test_pow_symm_gradient_double_eigs(self):
        val1 = 2.0
        val2 = 0.5
        C = self.R@np.diag(np.array([val1, val2, val1]))@self.R.T
        m = 3
        check_grads(lambda A: TensorMath.pow_symm(A, m), (C,), order=1)

    def test_pow_symm_gradient_distinct_eigenvalues(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        C = F.T@F
        m = 0.25
        check_grads(lambda A: TensorMath.pow_symm(C, m), (C,), order=1)

    def test_pow_symm_gradient_almost_double_degenerate(self):
        C = self.R@np.diag(np.array([2.1, 2.1 + 1e-8, 3.0]))@self.R.T
        m = 0.25
        check_grads(lambda A: TensorMath.pow_symm(A, 0.25), (C,), order=1, atol=1e-16, eps=1e-10)


    def test_determinant(self):
        A = np.array([[5/9, 4/7, 2/11],
                      [7/9, 4/9, 1/5],
                      [1/3, 3/7, 17/18]])
        self.assertEqual(TensorMath.det(A), -45583/280665)

    def test_detpIm1(self):
        A = np.array([[-8.7644781692191447986e-7, -0.00060943437636452272438, 0.0006160110345770283824],
                      [0.00059197095431573693372, -0.00032421698142571543644, -0.00075031460538177354586],
                      [-0.00057095032376313107833, 0.00042675236045286923589, -0.00029239794707394684004]])
        exact = -0.00061636368316760725654 # computed with exact arithmetic in Mathematica and truncated
        val = TensorMath.detpIm1(A)
        self.assertAlmostEqual(exact, val, 15)

    def test_determinant_precision(self):
        eps = 1e-8
        A = np.diag(np.array([eps, eps, eps]))
        # det(A + I) - 1
        exact = eps**3 + 3*eps**2 + 3*eps

        # straightforward approach loses precision
        Jm1 = TensorMath.det(A + np.identity(3)) - 1
        error = np.abs((Jm1 - exact)/exact)
        self.assertGreater(error, 1e-9)

        # special function retains precision
        Jm1 = TensorMath.detpIm1(A)
        error = np.abs((Jm1 - exact)/exact)
        self.assertEqual(error, 0)

    def test_right_polar_decomp(self):
        key = jax.random.PRNGKey(0)
        F = jax.random.uniform(key, (3,3), minval=1e-8, maxval=10.0)
        R, U = TensorMath.right_polar_decomposition(F)
        # R is orthogonal
        self.assertArrayNear(R@R.T, np.identity(3), 14)
        self.assertArrayNear(R.T@R, np.identity(3), 14)
        # U is symmetric
        self.assertArrayNear(U, TensorMath.sym(U), 14)
        # RU = F
        self.assertArrayNear(R@U, F, 14)


if __name__ == '__main__':
    unittest.main()
