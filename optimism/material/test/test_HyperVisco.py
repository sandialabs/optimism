import unittest

import jax
import jax.numpy as np
from jax.scipy import linalg
from matplotlib import pyplot as plt

from optimism.material import HyperViscoelastic as HyperVisco
from optimism.test.TestFixture import TestFixture
from optimism.material import MaterialUniaxialSimulator
from optimism.TensorMath import deviator
from optimism.TensorMath import log_symm

plotting = False

class HyperViscoModelFixture(TestFixture):
    def setUp(self):
    
        G_eq = 0.855 # MPa
        K_eq = 1000*G_eq # MPa
        G_neq_1 = 5.0
        # tau_1   = 0.1
        tau_1 = 25.0
        self.G_neq_1 = G_neq_1
        self.tau_1 = tau_1 # needed for analytic calculation below
        self.props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 

        materialModel = HyperVisco.create_material_model_functions(self.props)
        # do this on base fixture class to reduce jitting in test fixtures
        self.energy_density = jax.jit(materialModel.compute_energy_density)
        self.compute_state_new = jax.jit(materialModel.compute_state_new)
        self.compute_initial_state = materialModel.compute_initial_state
        self.compute_material_qoi = jax.jit(materialModel.compute_material_qoi)


# This test is covered from the others
# class HyperViscoZeroTest(HyperViscoModelFixture):
#     def test_zero_point(self):
#         dispGrad = np.zeros((3,3))
#         initialState = self.compute_initial_state()
#         dt = 1.0

#         energy = self.energy_density(dispGrad, initialState, dt)
#         self.assertNear(energy, 0.0, 12)

#         state = self.compute_state_new(dispGrad, initialState, dt)
#         self.assertArrayNear(state, np.eye(3).ravel(), 12)

#         dissipatedEnergy = self.compute_material_qoi(dispGrad, initialState, dt)
#         self.assertNear(dissipatedEnergy, 0.0, 12)

#     def test_regression_nonzero_point(self):
#         key = jax.random.PRNGKey(1)
#         dispGrad = jax.random.uniform(key, (3, 3))
#         initialState = self.compute_initial_state()
#         dt = 1.0

#         energy = self.energy_density(dispGrad, initialState, dt)
#         self.assertNear(energy, 133.4634466468207, 12)

#         state = self.compute_state_new(dispGrad, initialState, dt)
#         stateGold = np.array([0.988233534321, 0.437922586964, 0.433881277313, 
#                               0.437922586964, 1.378870045574, 0.079038974065, 
#                               0.433881277313, 0.079038974065, 1.055381729505])
#         self.assertArrayNear(state, stateGold, 12)

#         dissipatedEnergy = self.compute_material_qoi(dispGrad, initialState, dt)
#         self.assertNear(dissipatedEnergy, 0.8653744383204761, 12)


class HyperViscoUniaxialStrain(HyperViscoModelFixture):
    def test_loading_only(self):
        strain_rate = 1.0e-2
        total_time = 100.0
        n_steps = 100
        dt = total_time / n_steps
        times = np.linspace(0.0, total_time, n_steps)
        Fs = jax.vmap(
            lambda t: np.array(
                [[np.exp(strain_rate * t), 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]
            )
        )(times)
        state_old = self.compute_initial_state()
        energies = np.zeros(n_steps)
        states = np.zeros((n_steps, state_old.shape[0]))

        for n, F in enumerate(Fs):
            dispGrad = F - np.eye(3)
            energies = energies.at[n].set(self.energy_density(dispGrad, state_old, dt))
            state_new = self.compute_state_new(dispGrad, state_old, dt)
            states = states.at[n, :].set(state_new)
            state_old = state_new


        Fvs = jax.vmap(lambda Fv: Fv.reshape((3, 3)))(states)
        Fes = jax.vmap(lambda F, Fv: F @ np.linalg.inv(Fv), in_axes=(0, 0))(Fs, Fvs)

        Es  = jax.vmap(lambda F: log_symm(F))(Fs)
        Evs = jax.vmap(lambda Fv: log_symm(Fv))(Fvs)
        Ees = jax.vmap(lambda Fe: log_symm(Fe))(Fes)
        # print(Fs)

        e_v_11 = (2. / 3.) * strain_rate * times - \
                 (2. / 3.) * strain_rate * self.tau_1 * (1. - np.exp(-times / self.tau_1))

        e_e_11 = strain_rate * times - e_v_11
        e_e_22 = 0.5 * e_v_11

        Ee_analytic = jax.vmap(
            lambda e_11, e_22: np.array(
                [[e_11, 0., 0.],
                 [0., e_22, 0.],
                 [0., 0., e_22]]
            ), in_axes=(0, 0)
        )(e_e_11, e_e_22)
        Me_analytic = jax.vmap(lambda Ee: 2. * self.G_neq_1 * deviator(Ee))(Ee_analytic)
        Dv_analytic = jax.vmap(lambda Me: 1. / (2. * self.G_neq_1 * self.tau_1) * deviator(Me))(Me_analytic)
        dissipations_analytic = jax.vmap(lambda Me, Dv: np.tensordot(Me, Dv), in_axes=(0, 0))(Me_analytic, Dv_analytic)

        # calculate Mandel stress to compare dissipation
        Dvs = jax.vmap(lambda Ee: (1. / self.tau_1) * deviator(Ee))(Ees)
        Mes = jax.vmap(lambda Ee: 2. * self.G_neq_1 * deviator(Ee))(Ees)
        dissipations = jax.vmap(lambda D, M: np.tensordot(D, M), in_axes=(0, 0))(Dvs, Mes)


        self.assertArrayNear(Evs[:, 0, 0], e_v_11, 3)
        self.assertArrayNear(Ees[:, 0, 0], e_e_11, 3)
        self.assertArrayNear(Ees[:, 1, 1], e_e_22, 3)
        self.assertArrayNear(dissipations, dissipations_analytic, 3)

        if plotting:
            plt.figure(1)
            plt.plot(times, np.log(Fs[:, 0, 0]))
            plt.savefig('times_Fs.png')

            plt.figure(2)
            plt.plot(times, energies)
            plt.savefig('times_energies.png')

            plt.figure(3)
            plt.plot(times, Evs[:, 0, 0], marker='o', linestyle='None', markevery=10)
            plt.plot(times, e_v_11, linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('Viscous Logarithmic Strain 11 component')
            plt.savefig('times_viscous_stretch.png')

            plt.figure(4)
            plt.plot(times, Ees[:, 0, 0], marker='o', linestyle='None', markevery=10, label='11', color='blue')
            plt.plot(times, Ees[:, 1, 1], marker='o', linestyle='None', markevery=10, label='22', color='red')
            plt.plot(times, e_e_11, linestyle='--', color='blue')
            plt.plot(times, e_e_22, linestyle='--', color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Viscous Elastic Strain')
            plt.legend(loc='best')
            plt.savefig('times_elastic_stretch.png')

            plt.figure(5)
            plt.plot(times, dissipations, marker='o', linestyle='None', markevery=10)
            plt.plot(times, dissipations_analytic, linestyle='--')
            plt.savefig('times_dissipation.png')


class HyperViscoUniaxial(TestFixture):
    def setUp(self):
        G_eq = 0.855 # MPa
        K_eq = 1*G_eq # MPa - artificially low bulk modulus so that volumetric strain energy doesn't drown out the dissipation (which is deviator)
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
