import os
import unittest

import jax
import jax.numpy as np

from optimism import Math
from optimism.test import TestFixture

SUM_TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'ill_conditioned_sum_data.npz')
DOTPROD_TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'ill_conditioned_dot_product_data.npz')

class TestMathSum(TestFixture.TestFixture):

    def setUp(self):
        data = np.load(SUM_TESTDATA_FILENAME)
        self.a = data['a']
        self.exactDoublePrec = data['exactDP'] #-2.3746591162238294e-1

    def test_sum2_on_ill_conditioned_sum(self):
        s = Math.sum2(self.a)
        # note that we are checking for exact double precision equality!
        self.assertEqual(s, self.exactDoublePrec)

    def test_sum2_jitted_on_ill_conditioned_sum(self):
        func = jax.jit(Math.sum2)
        s = func(self.a)
        self.assertEqual(s, self.exactDoublePrec)

    def test_numpy_sum_fails_badly_on_ill_conditioned_sum(self):
        s = np.sum(self.a)
        # numpy fails to get even 4 decimal places correct
        self.assertNotAlmostEqual(s, self.exactDoublePrec, 4)

    def test_grad_on_sum2_works(self):
        v = np.array([0.9653214 , 0.22515893, 0.63302994, 0.29638183])
        g = jax.grad(Math.sum2)(v)
        exact = np.ones(v.shape)
        self.assertArrayNear(g, exact, 14)


class TestMathInnerProduct(TestFixture.TestFixture):

    def setUp(self):
        data = np.load(DOTPROD_TESTDATA_FILENAME)
        self.x = data['x']
        self.y = data['y']
        self.exactDoublePrec = data['exactDP']

    def test_dot2_on_ill_conditioned_inner_product(self):
        ip = Math.dot2(self.x, self.y)
        self.assertEqual(ip, self.exactDoublePrec)

    def test_numpy_dot_fails_badly_on_ill_conditioned_inner_product(self):
        ip = np.dot(self.x, self.y)
        self.assertNotAlmostEqual(ip, self.exactDoublePrec, 4)

    def test_jit_dot2_on_ill_conditioned_inner_product(self):
        func = jax.jit(Math.dot2)
        ip = func(self.x, self.y)
        self.assertEqual(ip, self.exactDoublePrec)

    def test_grad_on_dot2_works(self):
        v = np.array([0.9653214 , 0.22515893, 0.63302994, 0.29638183])
        w = np.array([0.87241435, 0.44302094, 0.27708054, 0.786595  ])
        g = jax.grad(Math.dot2)(v,w)
        self.assertArrayNear(g, w, 14)
    

if __name__ == '__main__':
    unittest.main()
