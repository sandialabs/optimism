from matplotlib import pyplot as plt
from jax import random
from scipy.spatial.transform import Rotation as R

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import Objective
from optimism.test import TestFixture
from optimism.test.MeshFixture import MeshFixture
from optimism.phasefield import PhaseFieldThreshold as Model
from optimism import SparseMatrixAssembler
from optimism import TensorMath
from optimism import Mesh


plotting=False


class PhaseFieldThresholdModelFixture(TestFixture.TestFixture):
    
    def setUp(self):
        self.E = 100.0
        self.nu = 0.321
        self.Gc = 40.0
        self.l = 1.0
        props = {'elastic modulus': self.E,
                 'poisson ratio': self.nu,
                 'critical energy release rate': self.Gc,
                 'regularization length': self.l,
                 'kinematics': 'large deformations'}
        self.model = Model.create_material_model_functions(props)
        self.flux_func = grad(self.model.compute_energy_density, (0,1,2))
        self.internalVariables = self.model.compute_initial_state()


    def test_zero_point(self):
        dispGrad = np.zeros((3,3))
        phase = 0.
        phaseGrad = np.zeros(3)

        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables)
        self.assertNear(energy, 0.0, 12)

        stress, phaseForce, phaseGradForce = self.flux_func(dispGrad, phase, phaseGrad, self.internalVariables)
        self.assertArrayNear(stress, np.zeros((3,3)), 12)
        self.assertNear(phaseForce, 3.0/8.0*self.Gc/self.l, 12)
        self.assertArrayNear(phaseGradForce, np.zeros(3), 12)


    def test_rotation_invariance(self):
        key = random.PRNGKey(0)
        dispGrad = random.uniform(key, (3,3))
        key, subkey = random.split(key)
        phase = random.uniform(subkey)
        key,subkey = random.split(key)
        phaseGrad = random.uniform(subkey, (3,))
        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables)
        
        Q = R.random(random_state=1234).as_matrix()
        dispGradStar = Q@(dispGrad + np.identity(3)) - np.identity(3)
        phaseStar = phase
        phaseGradStar = Q@phaseGrad
        internalVariablesStar = self.internalVariables
        energyStar = self.model.compute_energy_density(dispGradStar, phaseStar, phaseGradStar, internalVariablesStar)
        self.assertNear(energy, energyStar, 12)


    def test_uniaxial_energy(self):
        strain = 0.1 # engineering strain
        F = np.diag(np.exp(strain*np.array([1.0, -self.nu, -self.nu])))
        dispGrad =  F - np.identity(3)
        phase = 0.15
        phaseGrad = np.zeros(3)

        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables)

        g = (1.0 - phase)**2
        energyExact = g*0.5*self.E*strain**2 + 3/8*self.Gc/self.l*phase
        self.assertNear(energy, energyExact, 12)

        piolaStress,_,_ = self.flux_func(dispGrad, phase, phaseGrad, self.internalVariables)
        kStress = piolaStress@(dispGrad + np.identity(3)).T
        kStressExact = np.zeros((3,3)).at[0,0].set(g*self.E*strain)
        self.assertArrayNear(kStress, kStressExact, 12)


if __name__ == '__main__':
    TestFixture.unittest.main()
