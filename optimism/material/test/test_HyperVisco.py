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
        self.energy_density = jax.jit(materialModel.compute_energy_density)
        self.compute_state_new = jax.jit(materialModel.compute_state_new)
        self.compute_initial_state = materialModel.compute_initial_state
        self.compute_material_qoi = jax.jit(materialModel.compute_material_qoi)

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

        # numerical solution
        for n, F in enumerate(Fs):
            dispGrad = F - np.eye(3)
            energies = energies.at[n].set(self.energy_density(dispGrad, state_old, dt))
            state_new = self.compute_state_new(dispGrad, state_old, dt)
            states = states.at[n, :].set(state_new)
            state_old = state_new


        Fvs = jax.vmap(lambda Fv: Fv.reshape((3, 3)))(states)
        Fes = jax.vmap(lambda F, Fv: F @ np.linalg.inv(Fv), in_axes=(0, 0))(Fs, Fvs)

        Evs = jax.vmap(lambda Fv: log_symm(Fv))(Fvs)
        Ees = jax.vmap(lambda Fe: log_symm(Fe))(Fes)

        Dvs = jax.vmap(lambda Ee: (1. / self.tau_1) * deviator(Ee))(Ees)
        Mes = jax.vmap(lambda Ee: 2. * self.G_neq_1 * deviator(Ee))(Ees)
        dissipations = jax.vmap(lambda D, M: np.tensordot(D, M), in_axes=(0, 0))(Dvs, Mes)

        # analytic solution
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

        # test
        self.assertArrayNear(Evs[:, 0, 0], e_v_11, 3)
        self.assertArrayNear(Ees[:, 0, 0], e_e_11, 3)
        self.assertArrayNear(Ees[:, 1, 1], e_e_22, 3)
        self.assertArrayNear(dissipations, dissipations_analytic, 3)

        if plotting:
            plt.figure(1)
            plt.plot(times, np.log(Fs[:, 0, 0]))
            plt.xlabel('Time (s)')
            plt.ylabel('F 11')
            plt.savefig('times_Fs.png')

            plt.figure(2)
            plt.plot(times, energies)
            plt.xlabel('Time (s)')
            plt.ylabel('Algorithmic energy')
            plt.savefig('times_energies.png')

            plt.figure(3)
            plt.plot(times, Evs[:, 0, 0], marker='o', linestyle='None', markevery=10)
            plt.plot(times, e_v_11, linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('Viscous Logarithmic Strain 11 component')
            plt.legend(["Numerical", "Analytic"], loc=0, frameon=True)
            plt.savefig('times_viscous_stretch.png')

            plt.figure(4)
            plt.plot(times, Ees[:, 0, 0], marker='o', linestyle='None', markevery=10, label='11', color='blue')
            plt.plot(times, Ees[:, 1, 1], marker='o', linestyle='None', markevery=10, label='22', color='red')
            plt.plot(times, e_e_11, linestyle='--', color='blue')
            plt.plot(times, e_e_22, linestyle='--', color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Viscous Elastic Strain')
            plt.legend(["11 Numerical", "22 Numerical", "11 Analytic", "22 Analytic"], loc=0, frameon=True)
            plt.savefig('times_elastic_stretch.png')

            plt.figure(5)
            plt.plot(times, dissipations, marker='o', linestyle='None', markevery=10)
            plt.plot(times, dissipations_analytic, linestyle='--')
            plt.legend(["Numerical", "Analytic"], loc=0, frameon=True)
            plt.savefig('times_dissipation.png')



if __name__ == '__main__':
    unittest.main()
