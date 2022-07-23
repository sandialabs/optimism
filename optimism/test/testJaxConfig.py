import unittest

from optimism.JaxConfig import *
from optimism.test import TestFixture

from jax import config


class TestJaxConfiguration(TestFixture.TestFixture):

    def test_double_precision_mode_is_on(self):
        a = np.array([1.0])
        self.assertTrue(a.dtype == np.float64)

    def test_debug_nans_is_off(self):
        np.log(-1.0)

    def test_debug_infs_is_off(self):
        a = np.array([1.0])
        a/0

    def test_jit_is_enabled(self):
        self.assertFalse(config.jax_disable_jit)

if __name__ == "__main__":
    unittest.main()
