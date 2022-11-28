import unittest

import jax
import jax.numpy as np

from optimism.material import Gent
from optimism.test import TestFixture


class TestGentMaterial(TestFixture.TestFixture):
    def setUp(self):
        kappa = 100.0
        mu = 10.0
        Jm = 4.0
        
        properties = {"bulk modulus": kappa,
                      "shear modulus": mu,
                      "Jm parameter": Jm}
        
        self.material = Gent.create_material_functions(properties)
        
        self.internalVariables = self.material.compute_initial_state()
        
    def test_zero_point(self):
        dispGrad = np.zeros((3, 3))        
        W = self.material.compute_energy_density(dispGrad, self.internalVariables)
        self.assertLessEqual(W, np.linalg.norm(dispGrad)*1e-10)

if __name__ == "__main__":
    unittest.main()