import unittest
from matplotlib import pyplot as plt
from jax import random
from scipy.spatial.transform import Rotation as R

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import Objective
from optimism.test.TestFixture import TestFixture
from optimism.test.MeshFixture import MeshFixture
from optimism.phasefield import PhaseFieldThresholdPlastic as Model
from optimism import SparseMatrixAssembler
from optimism import TensorMath
from optimism import Mesh


plotting=True

flux_func = jacfwd(Model.energy_density, (0,1,2))


class GradOfPlasticPhaseFieldModelFixture(TestFixture):
    
    def setUp(self):
        E = 100.0
        poisson = 0.321
        Gc = 40.0
        l = 1.0
        Y0 = 0.3*E
        H = 1.0e-2*E      
        self.props = Model.make_properties(E, poisson, Gc, l, Y0, H)


    def test_zero_point(self):
        dispGrad = np.zeros((3,3))
        phase = 0.
        phaseGrad = np.zeros(3)
        state = Model.make_initial_state()

        energy = Model.energy_density(dispGrad, phase, phaseGrad, state, self.props)
        self.assertNear(energy, 0.0, 12)

        stress, phaseForce, phaseGradForce = flux_func(dispGrad, phase, phaseGrad, state, self.props)
        self.assertArrayNear(stress, np.zeros((3,3)), 12)
        self.assertNear(phaseForce, 3.0/8.0*self.props.Gc/self.props.l, 12)
        self.assertArrayNear(phaseGradForce, np.zeros(3), 12)


    def test_rotation_invariance(self):
        key = random.PRNGKey(0)
        dispGrad = random.uniform(key, (3,3))
        key, subkey = random.split(key)
        phase = random.uniform(subkey)
        key,subkey = random.split(key)
        phaseGrad = random.uniform(subkey, (3,))
        internalVariables = Model.make_initial_state()
        energy = Model.energy_density(dispGrad, phase, phaseGrad, internalVariables, self.props)
        
        Q = R.random(random_state=1234).as_matrix()
        dispGradStar = Q@dispGrad@Q.T
        phaseStar = phase
        phaseGradStar = Q@phaseGrad
        internalVariablesStar = internalVariables
        energyStar = Model.energy_density(dispGradStar, phaseStar, phaseGradStar, internalVariablesStar, self.props)
        self.assertNear(energy, energyStar, 12)


    def test_elastic_energy(self):
        strainBelowYield = 0.5*self.props.Y0/self.props.E
        dispGrad = strainBelowYield*np.diag(np.array([1.0, -self.props.nu, -self.props.nu]))
        phase = 0.0
        phaseGrad = np.zeros(3)
        state = Model.make_initial_state()

        energy = Model.energy_density(dispGrad, phase, phaseGrad, state, self.props)
        WExact = 0.5*self.props.E*strainBelowYield**2
        self.assertNear(energy, WExact, 12)

        stress,_,_ = flux_func(dispGrad, phase, phaseGrad, state, self.props)
        stressExact = np.zeros((3,3)).at[0,0].set(self.props.E*strainBelowYield)
        self.assertArrayNear(stress, stressExact, 12)


    def test_plastic_stress(self):
        huge = 1e10
        
        E = 100.0
        nu = 0.321
        Gc = huge
        l = 1.0
        Y0 = 0.3*E
        H = 1.0e-2*E
        props = Model.make_properties(E, nu, Gc, l, Y0, H)

        strainAboveYield = 1.25*self.props.Y0/self.props.E
        dispGrad = strainAboveYield*np.diag(np.array([1.0, -self.props.nu, -self.props.nu]))
        phase = 0.0
        phaseGrad = np.zeros(3)
        state = Model.make_initial_state()
        stress,_,_ = flux_func(dispGrad, phase, phaseGrad, state, self.props)
        print("stress=",stress)
        
        
    def no_test_plastic_strain_path(self):
        
        strain = np.zeros((3,3))
        dispGrad = np.zeros((3,3))
        strain_inc = 0.1
        stateOld = np.zeros((10,))

        energy_density = jit(J2.energy_density)
        stress_func = jit(grad(lambda elStrain, state, props: J2.energy_density(elStrain, state, props, doUpdate=False)))
        tangents_func = jit(hessian(J2.energy_density))
        compute_state_new = jit(J2.compute_state_new)
        
        strainHistory = []
        stressHistory = []
        tangentsHistory = []
        eqpsHistory = []
        energyHistory = []
        for i in range(10):
            energy  = energy_density(strain, stateOld, self.props)
            tangentsNew = tangents_func(strain, stateOld, self.props)
            stateNew = compute_state_new(strain, stateOld, self.props)
            stressNew = stress_func(strain, stateNew, self.props)
            strainHistory.append(strain[0,0])
            stressHistory.append(stressNew[0,0])
            tangentsHistory.append(tangentsNew[0,0,0,0])
            eqpsHistory.append(stateNew[J2.EQPS])
            energyHistory.append(energy)

            stateOld = stateNew
            strain = strain.at[0,0].add(strain_inc)
            dispGrad = strain

        if plotting:
            plt.figure()
            plt.plot(strainHistory, energyHistory, marker='o', fillstyle='none')
            plt.xlabel('strain')
            plt.ylabel('potential density')

            plt.savefig('energy.png')
            
            plt.figure()
            plt.plot(strainHistory, stressHistory, marker='o', fillstyle='none')
            plt.xlabel('strain')
            plt.ylabel('stress')

            plt.savefig('stress.png')
            
            plt.figure()
            plt.plot(strainHistory, eqpsHistory, marker='o')
            plt.xlabel('strain')
            plt.ylabel('Eq Pl Strain')

            plt.savefig('eqps.png')
            
            plt.figure()
            plt.plot(strainHistory, tangentsHistory, marker='o')
            plt.xlabel('strain')
            plt.ylabel('Tangent modulus')
            
            plt.savefig('tangent.png')
            
        E = self.props[J2.PROPS_E]
        mu = self.props[J2.PROPS_MU]
        nu = self.props[J2.PROPS_NU]
        lam = E*nu/(1.0 + nu)/(1.0 - 2.0*nu)
        Y0 = self.props[J2.PROPS_Y0]
        H = self.props[J2.PROPS_H]
        strainEnd = 0.9
        # for solutions, refer to jax-fem/papers/plane_strain_unit_test.pdf
        eqpsExp = [max(0.0, (2.0 * i * mu - Y0) / (3.0*mu + H) ) for i in strainHistory] 
        stressNewExp = (2.0 * mu * Y0 + \
            2.0 * strainEnd * pow(mu,2) + \
            strainEnd * H * lam + \
            2.0 * strainEnd * H * mu + \
            3.0 * strainEnd * lam * mu) \
            / (3.0 * mu + H)

        self.assertNear( stressHistory[-1], stressNewExp, 11)
        for i in range(0, len(strainHistory)):
            self.assertNear( eqpsHistory[i], eqpsExp[i], 12)

if __name__ == '__main__':
    unittest.main()
