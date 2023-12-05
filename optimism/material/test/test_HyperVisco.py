import unittest

import jax
import jax.numpy as np
from jax.scipy import linalg

from optimism.material import HyperViscoelastic as HyperVisco
from optimism.test.TestFixture import TestFixture

def make_disp_grad_from_strain(strain):
    return linalg.expm(strain) - np.identity(3)
        

class HyperViscoModelFixture(TestFixture):
    def setUp(self):
    
        G_eq = 0.855 # MPa
        K_eq = 1000*G_eq # MPa
        G_neq_1 = 5.0
        tau_1   = 0.1
        self.props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 

        materialModel = HyperVisco.create_material_model_functions(self.props)

        self.energy_density = jax.jit(materialModel.compute_energy_density)
        self.compute_state_new = materialModel.compute_state_new
        self.compute_initial_state = materialModel.compute_initial_state
        
    def test_zero_point(self):
        dispGrad = np.zeros((3,3))
        initialState = self.compute_initial_state()
        dt = 1.0

        energy = self.energy_density(dispGrad, initialState, dt)
        self.assertNear(energy, 0.0, 12)

        state = self.compute_state_new(dispGrad, initialState, dt)
        self.assertArrayNear(state, np.eye(3).ravel(), 12)

    def test_regression_nonzero_point(self):
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))
        initialState = self.compute_initial_state()
        dt = 1.0

        energy = self.energy_density(dispGrad, initialState, dt)
        self.assertNear(energy, 133.3469451269987, 12)

        state = self.compute_state_new(dispGrad, initialState, dt)
        stateGold = np.array([0.988233534321, 0.437922586964, 0.433881277313, 
                              0.437922586964, 1.378870045574, 0.079038974065, 
                              0.433881277313, 0.079038974065, 1.055381729505])
        self.assertArrayNear(state, stateGold, 12)

        
if __name__ == '__main__':
    unittest.main()
