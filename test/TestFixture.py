import unittest
import numpy

class TestFixture(unittest.TestCase):
    def assertArrayEqual(self, a, b):
        self.assertEqual( len(a), len(b) )
        self.assertIsNone(numpy.testing.assert_array_equal(a,b))

    def assertArrayNotEqual(self, a, b):
        self.assertTrue( (a-b != 0).any() )

    def assertArrayNear(self, a, b, decimals):
        self.assertEqual(len(a), len(b))
        self.assertIsNone(numpy.testing.assert_array_almost_equal(a,b,decimals))

    def assertNear(self, a, b, decimals):
        self.assertAlmostEqual(a, b, decimals)
