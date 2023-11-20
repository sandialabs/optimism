import unittest
from scipy.spatial.transform import Rotation

import jax
from jax import numpy as np
from jax.test_util import check_grads

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


class TensorMathFixture(TestFixture):

    def setUp(self):
        self.log_squared = lambda A: np.tensordot(TensorMath.log_sqrt(A), TensorMath.log_sqrt(A))
        

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

    def test_tensor_sqrt(self):
        eigvals = np.array([2., 0.5, 2.])
        C = R@np.diag(eigvals)@R.T
        U = TensorMath.mtk_sqrt(C)
        self.assertArrayNear(U, TensorMath.sym(U), 14)
        self.assertArrayNear(U@U, C, 14)


if __name__ == '__main__':
    unittest.main()
