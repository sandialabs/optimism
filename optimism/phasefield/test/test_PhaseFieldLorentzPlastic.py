from matplotlib import pyplot as plt
from jax import random
from scipy.spatial.transform import Rotation as R
import unittest

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import Objective
from optimism.test import TestFixture
from optimism.test.MeshFixture import MeshFixture
from optimism.phasefield import PhaseFieldLorentzPlastic as Model
from optimism import SparseMatrixAssembler
from optimism import TensorMath
from optimism import Mesh


plotting=False


class GradOfPlasticPhaseFieldModelFixture(TestFixture.TestFixture):
    
    def setUp(self):
        self.E = 100.0
        self.nu = 0.321
        self.Gc = 40.0
        self.psiC = 0.5*self.E
        self.l = 1.0
        self.Y0 = 0.3*self.E
        self.H = 1.0e-2*self.E
        props = {'elastic modulus': self.E,
                 'poisson ratio': self.nu,
                 'critical energy release rate': self.Gc,
                 'critical strain energy density': self.psiC,
                 'regularization length': self.l,
                 'yield strength': self.Y0,
                 'hardening model': 'linear',
                 'hardening modulus': self.H,
                 'kinematics': 'large deformations'}
        self.model = Model.create_material_model_functions(props)
        self.flux_func = grad(self.model.compute_energy_density, (0,1,2))
        self.internalVariables = self.model.compute_initial_state()
        self.dt = 1.0


    def test_zero_point(self):
        dispGrad = np.zeros((3,3))
        phase = 0.
        phaseGrad = np.zeros(3)

        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
        self.assertNear(energy, 0.0, 12)

        stress, phaseForce, phaseGradForce = self.flux_func(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
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
        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
        
        Q = R.random(random_state=1234).as_matrix()
        dispGradStar = Q@(dispGrad + np.identity(3)) - np.identity(3)
        phaseStar = phase
        phaseGradStar = Q@phaseGrad
        internalVariablesStar = self.internalVariables
        energyStar = self.model.compute_energy_density(dispGradStar, phaseStar, phaseGradStar, internalVariablesStar, self.dt)
        self.assertNear(energy, energyStar, 12)


    def test_elastic_energy(self):
        strainBelowYield = 0.5*self.Y0/self.E # engineering strain
        dispGrad = np.diag(np.exp(strainBelowYield*np.array([1.0, -self.nu, -self.nu])))-np.identity(3)
        phase = 0.0
        phaseGrad = np.zeros(3)

        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
        energyExact = 0.5*self.E*strainBelowYield**2
        self.assertNear(energy, energyExact, 12)

        piolaStress,_,_ = self.flux_func(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
        mandelStress = piolaStress@(dispGrad + np.identity(3)).T
        stressExact = np.zeros((3,3)).at[0,0].set(self.E*strainBelowYield)
        self.assertArrayNear(mandelStress, stressExact, 12)


    def test_plastic_stress(self):
    
        strain11 = 1.1*self.Y0/self.E
        eqps = (self.E*strain11 - self.Y0)/(self.H + self.E)
        elasticStrain11 = strain11 - eqps
        lateralStrain = -self.nu*elasticStrain11 - 0.5*eqps
        strains = np.array([strain11, lateralStrain, lateralStrain])
        dispGrad = np.diag(np.exp(strains)) - np.identity(3)
        phase = 0.0
        phaseGrad = np.zeros(3)

        energyExact = 0.5*self.E*elasticStrain11**2 + self.Y0*eqps + 0.5*self.H*eqps**2
        energy = self.model.compute_energy_density(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
        self.assertNear(energy, energyExact, 12)
        
        stress,_,_ = self.flux_func(dispGrad, phase, phaseGrad, self.internalVariables, self.dt)
        mandelStress = stress@(dispGrad + np.identity(3)).T
        mandelStress11Exact = self.E*(strain11 - eqps)
        self.assertNear(mandelStress[0,0], mandelStress11Exact, 12)      
        

if __name__ == '__main__':
    unittest.main()
