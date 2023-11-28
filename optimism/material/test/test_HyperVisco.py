import unittest

import jax
import jax.numpy as np
from jax.scipy import linalg

from optimism.material import HyperViscoelastic as HyperVisco
from optimism.test.TestFixture import TestFixture

from optimism import TensorMath

def make_disp_grad_from_strain(strain):
    return linalg.expm(strain) - np.identity(3)
        

class GradOfPlasticityModelFixture(TestFixture):
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
        self.assertNear(energy, 193.6029283822052, 12)

        state = self.compute_state_new(dispGrad, initialState, dt)
        stateGold = np.array([0.946976197737, 0.206631668313, 0.220846029827, 
                              0.206631668313, 1.155826112068, 0.015472488047, 
                              0.220846029827, 0.015472488047, 1.003179626352])
        self.assertArrayNear(state, stateGold, 12)

    def test_log_strain_updates(self):
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))
        initialState = self.compute_initial_state()
        dt = 1.0

        props = np.array([
            self.props['equilibrium bulk modulus'],
            self.props['equilibrium shear modulus'],
            self.props['non equilibrium shear modulus'],
            self.props['relaxation time']
        ])

        Ee_trial = HyperVisco._compute_elastic_logarithmic_strain(dispGrad, initialState)
        state_inc = HyperVisco._compute_state_increment(Ee_trial, dt, props)

        Ee_new_way = Ee_trial - state_inc # doesn't work

        # old implementation
        F = dispGrad + np.identity(3)
        Fv_old = initialState.reshape((3, 3))
        Fv_new = linalg.expm(state_inc)@Fv_old
        Fe = F @ np.linalg.inv(Fv_new)
        Ee_old_way = TensorMath.mtk_log_sqrt(Fe.T @ Fe)

        self.assertArrayNear(Ee_new_way.ravel(), Ee_old_way.ravel(), 12)

        
if __name__ == '__main__':
    unittest.main()
