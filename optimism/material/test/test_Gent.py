import numpy as onp
from scipy.spatial.transform import Rotation
import unittest

import jax
import jax.numpy as np

from optimism.material import Gent
from optimism.test import TestFixture


class TestGentMaterial(TestFixture.TestFixture):
    def setUp(self):
        self.kappa = 100.0
        self.mu = 10.0
        self.Jm = 4.0 # pick large enough to avoid singularity in tests
        
        properties = {"bulk modulus": self.kappa,
                      "shear modulus": self.mu,
                      "Jm parameter": self.Jm}
        
        self.material = Gent.create_material_functions(properties)
        
        self.internalVariables = self.material.compute_initial_state()

        self.dt = 0.0

    
    def test_zero_point(self):
        dispGrad = np.zeros((3, 3))
        W = self.material.compute_energy_density(dispGrad, self.internalVariables, self.dt)
        self.assertLessEqual(W, np.linalg.norm(dispGrad)*1e-10)


    def test_frame_indifference(self):
        # generate a random displacement gradient
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))
        

        W = self.material.compute_energy_density(dispGrad, self.internalVariables, self.dt)
        for i in range(10):
            Q = Rotation.random(random_state=i).as_matrix()
            dispGradTransformed = Q@(dispGrad + np.identity(3)) - np.identity(3)
            WStar = self.material.compute_energy_density(dispGradTransformed, self.internalVariables, self.dt)
            self.assertAlmostEqual(W, WStar, 12)
        
    
    def test_correspondence_with_linear_elasticity(self):
        zero = np.zeros((3, 3))
        C = jax.hessian(self.material.compute_energy_density)(zero, self.internalVariables, self.dt)
        
        lam = self.kappa - 2/3*self.mu
        
        CLinear = onp.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        CLinear[i, j, k, l] += self.mu if i == k and j == l else 0
                        CLinear[i, j, k, l] += self.mu if i == l and j == k else 0
                        CLinear[i, j, k, l] += lam if i == j and k == l else 0
        
        self.assertArrayNear(C, CLinear, 12)
    
    
    def test_finite_extensibility(self):
        # incompressible uniaxial extension
        # find stretch such that the strain energy just reaches the singularity.
        lockStretchCandidates = onp.roots([1.0, 0.0, -(self.Jm + 3), 2.0])
        lockStretch = onp.max(lockStretchCandidates)
        stretch = lockStretch*(1 + 1e-6) # account for finite precision of root finder
        I1 = stretch**2 + 2/stretch
        self.assertGreater(I1 - 3, self.Jm)
        
        # Check that energy is indeed infinite
        # (actually nan, since it produces a negative argument to a logarithm)
        F = np.diag(np.array([stretch, 1/np.sqrt(stretch), 1/np.sqrt(stretch)]))
        W = self.material.compute_energy_density(F - np.identity(3), self.internalVariables, self.dt)
        self.assertTrue(np.isnan(W))
        
        stretch = lockStretch*(1 - 1e-6)
        F = np.diag(np.array([stretch, 1/np.sqrt(stretch), 1/np.sqrt(stretch)]))
        W = self.material.compute_energy_density(F - np.identity(3), self.internalVariables, self.dt)
        self.assertFalse(np.isnan(W))

if __name__ == "__main__":
    unittest.main()