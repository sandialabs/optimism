from optimism.SmoothFunctions import *
from jax import grad
from optimism.test.TestFixture import TestFixture, unittest


class TestSmoothFunctions(TestFixture):

    def test_min(self):
        tol = 0.2
        eps = 1e-2
        tolm = tol*(1.-eps)
        
        x = 0.41

        self.assertEqual(x, min(x, x+tol, tol))
        self.assertEqual(x, min(x, x+1.1*tol, tol))
        self.assertEqual(x-tol, min(x, x-tol, tol))
        self.assertEqual(x-1.1*tol, min(x, x-1.1*tol, tol))

        self.assertEqual(x, min(x+tol, x, tol))
        self.assertEqual(x, min(x+1.1*tol, x, tol))
        self.assertEqual(x-tol, min(x-tol, x, tol))
        self.assertEqual(x-1.1*tol, min(x-1.1*tol, x, tol))
        
        minVal = min(x, x, tol)
        self.assertTrue(minVal > x - tol)
        self.assertTrue(minVal < x)

        minVal = min(x, x-tolm, tol)
        self.assertTrue(minVal > x - tol)
        self.assertTrue(minVal < x)

        minVal = min(x, x+tolm, tol)
        self.assertTrue(minVal > x - tol)
        self.assertTrue(minVal < x)

    def test_inf_min(self):
        eps = 1e-5

        self.assertEqual(0.0, min(0.0, np.inf, eps))
        self.assertEqual(0.0, min(np.inf, 0.0, eps))
        self.assertEqual(np.inf, min(np.inf, np.inf, eps))
        self.assertEqual(-np.inf, min(np.inf, -np.inf, eps))
        self.assertEqual(-np.inf, min(-np.inf, np.inf, eps))
        self.assertEqual(-np.inf, min(-np.inf, -np.inf, eps))

    def test_inf_grad_min(self):
        eps = 1e-5

        grad_min = grad(min, (0,1))
        
        self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(0.0, np.inf, eps))
        self.assertArrayEqual(np.array([0.0, 1.0]), grad_min(np.inf, 0.0, eps))
        #self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(np.inf, np.inf, eps))
        self.assertArrayEqual(np.array([0.0, 1.0]), grad_min(np.inf, -np.inf, eps))
        self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(-np.inf, np.inf, eps))
        #self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(-np.inf, -np.inf, eps))



if __name__ == '__main__':
    unittest.main()
