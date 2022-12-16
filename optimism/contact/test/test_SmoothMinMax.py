import matplotlib.pyplot as plt

from optimism.JaxConfig import *
from optimism.contact.SmoothMinMax import *
from optimism.test.TestFixture import *


class TestSmoothMinMax(TestFixture):

    def test_max_x_zero(self):
        tol = 0.2
        eps = 1e-15
        tolm = tol*(1.-eps)
        
        self.assertEqual(0.0, zmax(-tol, tol))
        self.assertNear(0.0, zmax(-tolm, tol), 15)
        
        self.assertEqual(tol, zmax(tol, tol))
        self.assertNear(tolm, zmax(tolm, tol), 15)
        self.assertEqual(1.1*tol, zmax(1.1*tol, tol))

        zmax_grad = grad( partial(zmax, eps=tol) )
        
        self.assertNear(zmax_grad(-eps), zmax_grad(eps), 14)
        self.assertNear(zmax_grad(-tol-eps), zmax_grad(-tol+eps), 14)
        self.assertNear(zmax_grad(tol-eps), zmax_grad(tol+eps), 14)
        
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
        self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(np.inf, np.inf, eps))
        self.assertArrayEqual(np.array([0.0, 1.0]), grad_min(np.inf, -np.inf, eps))
        self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(-np.inf, np.inf, eps))
        self.assertArrayEqual(np.array([1.0, 0.0]), grad_min(-np.inf, -np.inf, eps))



if __name__ == '__main__':
    unittest.main()
