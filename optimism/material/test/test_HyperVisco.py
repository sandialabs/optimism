import unittest

import jax
import jax.numpy as np
from jax.scipy import linalg

from optimism.material import HyperViscoelastic as HyperVisco
from optimism.test.TestFixture import TestFixture

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


    # def test_elastic_energy(self):
    #     strainBelowYield = 0.5*self.props['yield strength']/self.props['elastic modulus']
        
    #     strain = strainBelowYield*np.diag(np.array([1.0, -self.props['poisson ratio'], -self.props['poisson ratio']]))
    #     dispGrad = make_disp_grad_from_strain(strain)
    #     dt = 1.0
        
    #     state = self.compute_initial_state()

    #     energy = self.energy_density(dispGrad, state, dt)
    #     WExact = 0.5*self.props['elastic modulus']*strainBelowYield**2
    #     self.assertNear(energy, WExact, 12)

    #     F = dispGrad + np.identity(3)
    #     kirchhoffStress = self.stress_func(dispGrad, state, dt) @ F.T
    #     kirchhoffstressExact = np.zeros((3,3)).at[0,0].set(self.props['elastic modulus']*strainBelowYield)
    #     self.assertArrayNear(kirchhoffStress, kirchhoffstressExact, 12)

        
if __name__ == '__main__':
    unittest.main()
