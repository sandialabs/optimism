# Most of these tests are marked as expected failures. The reason: By default,
# jax enables a flag for XLA named "--xla_cpu_enable_fast_math=true", when
# executing on CPU. This ultimately sets a similar flag for LLVM. See here: 
# https://llvm.org/docs/LangRef.html#fast-math-flags
# The problem is that this flag allows associative changes to math expressions,
# which undermines the compensated sum and inner product algorithms.
# This flag can be turned off, but it kshould be evaluated for performance
# regression.

import os
# This is the method for turning off the flag.
# It should be executed before anything else (just before jax is
# imported)
# os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"

import unittest

from optimism.JaxConfig import *
from optimism import Math
from optimism.test import TestFixture

SUM_TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'ill_conditioned_sum_data.npz')
DOTPROD_TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'ill_conditioned_dot_product_data.npz')

class TestMathSum(TestFixture.TestFixture):

    def setUp(self):
        data = np.load(SUM_TESTDATA_FILENAME)
        self.a = data['a']
        self.exactDoublePrec = data['exactDP'] #-2.3746591162238294e-1


    @unittest.expectedFailure
    def test_sum2_on_ill_conditioned_sum(self):
        s = Math.sum2(self.a)
        # note that we are checking for exact double precision equality!
        self.assertEqual(s, self.exactDoublePrec)


    @unittest.expectedFailure
    def test_sum2_jitted_on_ill_conditioned_sum(self):
        func = jit(Math.sum2)
        s = func(self.a)
        self.assertEqual(s, self.exactDoublePrec)


    def test_numpy_sum_fails_badly_on_ill_conditioned_sum(self):
        s = np.sum(self.a)
        # numpy fails to get even 4 decimal places correct
        self.assertNotAlmostEqual(s, self.exactDoublePrec, 4)


    def test_grad_on_sum2_works(self):
        v = np.array([0.9653214 , 0.22515893, 0.63302994, 0.29638183])
        g = grad(Math.sum2)(v)
        exact = np.ones(v.shape)
        self.assertArrayNear(g, exact, 14)


class TestMathInnerProduct(TestFixture.TestFixture):

    def setUp(self):
        data = np.load(DOTPROD_TESTDATA_FILENAME)
        self.x = data['x']
        self.y = data['y']
        self.exactDoublePrec = data['exactDP']

        
    @unittest.expectedFailure
    def test_dot2_on_ill_conditioned_inner_product(self):
        ip = Math.dot2(self.x, self.y)
        self.assertEqual(ip, self.exactDoublePrec)


    def test_numpy_dot_fails_badly_on_ill_conditioned_inner_product(self):
        ip = np.dot(self.x, self.y)
        self.assertNotAlmostEqual(ip, self.exactDoublePrec, 4)


    @unittest.expectedFailure
    def test_jit_dot2_on_ill_conditioned_inner_product(self):
        func = jit(Math.dot2)
        ip = func(self.x, self.y)
        self.assertEqual(ip, self.exactDoublePrec)


    def test_grad_on_dot2_works(self):
        v = np.array([0.9653214 , 0.22515893, 0.63302994, 0.29638183])
        w = np.array([0.87241435, 0.44302094, 0.27708054, 0.786595  ])
        g = grad(Math.dot2)(v,w)
        self.assertArrayNear(g, w, 14)
    

if __name__ == '__main__':
    unittest.main()
