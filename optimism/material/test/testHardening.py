import unittest

from optimism.JaxConfig import *
from optimism.material import J2Plastic
from optimism.material import Hardening
from optimism.test.TestFixture import TestFixture


E = 100.0
nu = 0.321
Y0 = E*0.003

class VoceHardeningTestFixture(TestFixture):
    def setUp(self):
        self.Ysat = 4.0*Y0
        self.eps0 = 0.05
        self.props = {'hardening model': 'voce',
                      'elastic modulus': E,
                      'poisson ratio': nu,
                      'yield strength': Y0,
                      'saturation strength': self.Ysat,
                      'reference plastic strain': self.eps0}
        self.plasticEnergy, self.flowStrength = Hardening.create_hardening_model(self.props)

        
    def test_voce_hardening_zero_point(self):
        eqps = 0.0
        Wp = self.plasticEnergy(eqps, eqpsOld=0.0, dt=0.0)
        self.assertLess(Wp, 1e-14)


    def test_voce_hardening_yield_strength(self):
        eqps = 0.0
        Y = self.flowStrength(eqps, eqpsOld=0.0, dt=0.0)
        self.assertAlmostEqual(Y, Y0, 12)


    def test_voce_hardening_saturates_to_correct_value(self):
        eqps = 15.0*self.eps0
        Y = self.flowStrength(eqps, eqpsOld=0.0, dt=0.0)
        self.assertAlmostEqual(Y, self.Ysat, 5)


        
class PowerLawHardeningTestFixture(TestFixture):
    def setUp(self):
        self.n = 4.0
        self.eps0 = 0.05
        self.props = {'hardening model': 'power law',
                      'elastic modulus': E,
                      'poisson ratio': nu,
                      'yield strength': Y0,
                      'hardening exponent': self.n,
                      'reference plastic strain': self.eps0}
        self.plasticEnergy, self.flowStrength = Hardening.create_hardening_model(self.props)

        
    def test_power_law_hardening_zero_point(self):
        Wp = self.plasticEnergy(eqps=0.0, eqpsOld=0.0, dt=0.0)
        self.assertLess(Wp, 1e-14)


    def test_power_law_hardening_yield_strength(self):
        eqps = 0.0
        Y = self.flowStrength(eqps, eqpsOld=0.0, dt=0.0)
        self.assertAlmostEqual(Y, Y0, 12)


    def test_power_law_strength_increases(self):
        eqps = 5.0*self.eps0
        Y = self.flowStrength(eqps, eqpsOld=0.0, dt=0.0)
        self.assertGreater(Y, Y0)


    def test_power_law_hardening_slope_is_finite_at_origin(self):
        hardeningRate = jacfwd(self.flowStrength)
        eqps = 0.0
        H = hardeningRate(eqps, eqpsOld=0.0, dt=0.0)
        self.assertTrue(np.isfinite(H))


if __name__ == '__main__':
    unittest.main()
