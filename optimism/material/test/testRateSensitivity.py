import unittest

import jax

from optimism.material import Hardening
from optimism.test.TestFixture import TestFixture

class RateSensitivityFixture(TestFixture):
    def test_power_law_scales_correctly(self):
        S = 10.0
        m = 2.0
        epsDot0 = 1e-3
        
        compute_overstress = jax.grad(Hardening.power_law_rate_sensitivity)
        
        eqpsDot1 = 2.5
        dt = 1.5e-2
        eqpsOld = 0.37
        eqps1 = eqpsOld + eqpsDot1*dt
        stress1 = compute_overstress(eqps1, eqpsOld, dt, S, m, epsDot0)
        
        eqpsDot2 = eqpsDot1*2.0
        eqps2 = eqpsOld + eqpsDot2*dt
        stress2 = compute_overstress(eqps2, eqpsOld, dt, S, m, epsDot0)
        
        self.assertNear(stress2/stress1, (eqpsDot2/eqpsDot1)**(1/m), 10)


    def test_property_parsing(self):
        S = 10.0
        m = 1.0
        epsDot0 = 0.1
        Y0 = 3.0
        props = {"rate sensitivity": "power law",
                 "rate sensitivity exponent": m,
                 "rate sensitivity stress": S,
                 "reference plastic strain rate": epsDot0,
                 "hardening model": "linear",
                 "hardening modulus": 0.0,
                 "yield strength": Y0}
        
        hardening = Hardening.create_hardening_model(props)
        
        # viscous overstress will increase flow stress above initial yield
        eqps = 2.0
        eqpsOld = 1.0
        dt = 1.0e1
        flowStress = hardening.compute_flow_stress(eqps, eqpsOld, dt)
        self.assertGreater(flowStress, Y0)
        
        # if the time period is huge, the strain rate is so small that the
        # overstress should be lost in the floating point truncation error
        dt = 1e25
        flowStress = hardening.compute_flow_stress(eqps, eqpsOld, dt)
        self.assertEqual(flowStress, Y0)


if __name__ == "__main__":
    unittest.main()