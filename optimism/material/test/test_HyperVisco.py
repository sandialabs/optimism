import unittest

import jax
import jax.numpy as np
from jax.scipy import linalg
from matplotlib import pyplot as plt

from optimism.material import HyperViscoelastic as HyperVisco
from optimism.test.TestFixture import TestFixture
from optimism.material import MaterialUniaxialSimulator

plotting=True

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
        self.compute_material_qoi = materialModel.compute_material_qoi
        
    def test_zero_point(self):
        dispGrad = np.zeros((3,3))
        initialState = self.compute_initial_state()
        dt = 1.0

        energy = self.energy_density(dispGrad, initialState, dt)
        self.assertNear(energy, 0.0, 12)

        state = self.compute_state_new(dispGrad, initialState, dt)
        self.assertArrayNear(state, np.eye(3).ravel(), 12)

        dissipatedEnergy = self.compute_material_qoi(dispGrad, initialState, dt)
        self.assertNear(dissipatedEnergy, 0.0, 12)

    def test_regression_nonzero_point(self):
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))
        initialState = self.compute_initial_state()
        dt = 1.0

        energy = self.energy_density(dispGrad, initialState, dt)
        self.assertNear(energy, 133.4634466468207, 12)

        state = self.compute_state_new(dispGrad, initialState, dt)
        stateGold = np.array([0.988233534321, 0.437922586964, 0.433881277313, 
                              0.437922586964, 1.378870045574, 0.079038974065, 
                              0.433881277313, 0.079038974065, 1.055381729505])
        self.assertArrayNear(state, stateGold, 12)

        dissipatedEnergy = self.compute_material_qoi(dispGrad, initialState, dt)
        self.assertNear(dissipatedEnergy, 0.8653744383204761, 12)

class HyperViscoUniaxial(TestFixture):

    def setUp(self):
        G_eq = 0.855 # MPa
        K_eq = 1*G_eq # MPa - artificially low bulk modulus so that volumetric strain energy doesn't drown out the dissipation (which is deviatoric)
        G_neq_1 = 5.0
        tau_1   = 0.1
        self.props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        self.mat = HyperVisco.create_material_model_functions(self.props)

        self.totalTime = 1.0e1*tau_1

    def test_dissipated_energy_cycle(self):
        # cyclic loading
        stages = 2
        maxStrain = 2.0e-2
        maxTime = self.totalTime
        steps = 100
        strainInc = maxStrain/(steps-1)
        dt=maxTime/(steps-1)

        maxTime *= stages
        steps *= stages

        t0 = self.totalTime
        def cyclic_constant_true_strain_rate(t):
            outval = []
            for tval in t:
                if tval <= t0:
                    outval.append( (tval / dt) * strainInc )
                else:
                    outval.append( ( (maxTime - tval) / dt) * strainInc )
            return np.array(outval)

        uniaxial = MaterialUniaxialSimulator.run(self.mat, cyclic_constant_true_strain_rate, maxTime, steps=steps, tol=1e-8)

        # compute internal energy 
        intEnergies = []
        intEnergies.append(0.0)
        for step in range(1,steps): 
            F_diff = uniaxial.strainHistory[step] - uniaxial.strainHistory[step-1]
            P_sum = uniaxial.stressHistory[step] + uniaxial.stressHistory[step-1]
            intEnergies.append(0.5 * np.tensordot(P_sum,F_diff))
        internalEnergyHistory = np.cumsum(np.array(intEnergies))

        # compute free energy
        hyperVisco_props = HyperVisco._make_properties(self.props)
        def free_energy(dispGrad, state, dt):
            W_eq  = HyperVisco._eq_strain_energy(dispGrad, hyperVisco_props)

            Ee_trial = HyperVisco._compute_elastic_logarithmic_strain(dispGrad, state)
            delta_Ev = HyperVisco._compute_state_increment(Ee_trial, dt, hyperVisco_props)
            Ee = Ee_trial - delta_Ev 

            W_neq = HyperVisco._neq_strain_energy(Ee, hyperVisco_props)

            return W_eq + W_neq

        compute_free_energy = jax.jit(free_energy)
        freeEnergies = []
        freeEnergies.append(0.0)
        for step in range(1,steps): 
            freeEnergies.append( compute_free_energy(uniaxial.strainHistory[step], uniaxial.internalVariableHistory[step-1], dt) )
        freeEnergyHistory = np.array(freeEnergies)
        
        # compute dissipated energy
        timePoints = np.linspace(0.0, maxTime, num=steps)
        dt = timePoints[1] - timePoints[0]
        dissipatedEnergies = []
        dissipatedEnergies.append(0.0)
        compute_dissipation = jax.jit(self.mat.compute_material_qoi)
        for step in range(1,steps): 
            dissipatedEnergies.append( compute_dissipation(uniaxial.strainHistory[step], uniaxial.internalVariableHistory[step-1], dt) )
        dissipatedEnergyHistory = np.cumsum(np.array(dissipatedEnergies))
        
        # plot energies
        if plotting:
            plt.figure()
            plt.plot(uniaxial.time, internalEnergyHistory, marker='o', fillstyle='none')
            plt.plot(uniaxial.time, freeEnergyHistory, marker='x', fillstyle='none')
            plt.plot(uniaxial.time, dissipatedEnergyHistory, marker='v', fillstyle='none')
            plt.plot(uniaxial.time, freeEnergyHistory + dissipatedEnergyHistory, marker='s', fillstyle='none')

            plt.xlabel('Time')
            plt.ylabel('Energy Density (MPa)')
            plt.legend(["Internal", "Free", "Dissipated", "Free + Dissipated"], loc=0, frameon=True)
            plt.savefig('energy_histories.png')

        self.assertArrayNear(dissipatedEnergyHistory, internalEnergyHistory - freeEnergyHistory, 5)

        
if __name__ == '__main__':
    unittest.main()
