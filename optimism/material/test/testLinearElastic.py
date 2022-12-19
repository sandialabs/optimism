from scipy.spatial.transform import Rotation
import unittest

import jax
import jax.numpy as np

from optimism.material import LinearElastic
from optimism.test import TestFixture

class TestLinearElasticMaterial(TestFixture.TestFixture):
    def setUp(self):
        self.E = 10.0
        self.nu = 0.25
        
        properties = {"elastic modulus": self.E,
                      "poisson ratio": self.nu,
                      "strain measure": "logarithmic"}
        
        self.material = LinearElastic.create_material_model_functions(properties)
        
        self.internalVariables = self.material.compute_initial_state()
        self.dt = 0.0


    def test_zero_point(self):
        dispGrad = np.zeros((3, 3))
        W = self.material.compute_energy_density(dispGrad, self.internalVariables, self.dt)
        self.assertLessEqual(W, np.linalg.norm(dispGrad)*1e-10)

    
    def test_finite_deformation_frame_indifference(self):
        # generate a random displacement gradient
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))
        
        W = self.material.compute_energy_density(dispGrad, self.internalVariables, self.dt)
        for i in range(10):
            Q = Rotation.random(random_state=i).as_matrix()
            dispGradTransformed = Q@(dispGrad + np.identity(3)) - np.identity(3)
            WStar = self.material.compute_energy_density(dispGradTransformed, self.internalVariables, self.dt)
            self.assertAlmostEqual(W, WStar, 12)


    def test_internal_state_update(self):
        # generate a random displacement gradient
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))
        
        internalVariablesNew = self.material.compute_state_new(dispGrad, self.internalVariables, self.dt)
        # linear elastic has no internal variables
        self.assertEqual(internalVariablesNew.size, 0)


if __name__ == "__main__":
    unittest.main()