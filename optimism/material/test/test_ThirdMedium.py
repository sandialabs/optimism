import unittest
from optimism.test.TestFixture import TestFixture
import jax
import jax.numpy as np
from optimism.material import Neohookean
from optimism.material import ThirdMediumNeoHookean

from matplotlib import pyplot as plt

plotting = True

def energy_neo_hookean_adagio(dispGrad, kappa, mu):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    Wvol = 0.5*kappa*(0.5*J**2 - 0.5 - np.log(J))
    Wdev = 0.5*mu*(I1Bar - 3.0)
    return Wdev + Wvol

def energy_neo_hookean_rokos_volumetric(dispGrad, kappa, mu):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    return kappa/2.0 * np.log(J)**2

def energy_neo_hookean_rokos_isochoric(dispGrad, kappa, mu):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    return mu / 2.0 * (I1Bar - 3.0)

def energy_stable_neo_hookean(dispGrad, kappa, mu):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    I1 = np.tensordot(F,F)
    Wiso = mu/2.0*(I1 - 3.0)
    lam = kappa - (2.0/3.0) * mu
    alpha = 1.0 + mu/lam
    Wvol = lam/2.0 * (J - alpha)**2
    return Wiso + Wvol

def energy_stable_arruda_boyce(dispGrad, kappa, mu):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    I1 = np.tensordot(F,F)
    beta1 = 1.0
    beta2 = 1.0
    Wiso = mu/2.0*(I1 - 3.0) + beta1*mu/4.0*(I1**2 - 9.0) + beta2*mu/6.0*(I1**3 - 27.0)
    lam = kappa - (2.0/3.0) * mu
    alpha = 1.0 + mu/lam * (1.0 + 3.0*beta1 + 9.0*beta2)
    Wvol = lam/2.0 * (J - alpha)**2
    return Wiso + Wvol

def energy_teran_invertible(dispGrad, kappa, mu):
    lamda = kappa - (2.0/3.0) * mu
    props = np.array([mu, kappa, lamda])
    return ThirdMediumNeoHookean.teran_invertible_energy_density(dispGrad, props)

class ThirdMediumModelFixture(TestFixture):
    def setUp(self):
        self.kappa = 100.0
        self.mu = 10.0
        
    def test_energy_consistency(self):
        # generate a random displacement gradient
        key = jax.random.PRNGKey(1)
        dispGrad = jax.random.uniform(key, (3, 3))

        # compare to Neo-Hookean model
        props = {
            'elastic modulus': 9.0*self.kappa*self.mu / (3.0*self.kappa + self.mu),
            'poisson ratio': (3.0*self.kappa - 2.0*self.mu) / 2.0 / (3.0*self.kappa + self.mu),
            'version': 'adagio'
        }
        ref_material = Neohookean.create_material_model_functions(props)
        energy_gold = ref_material.compute_energy_density(dispGrad, ref_material.compute_initial_state(), dt=0.0)

        energy = energy_neo_hookean_adagio(dispGrad, self.kappa, self.mu)
        self.assertAlmostEqual(energy, energy_gold, 12)

    def test_plot_biaxial_response(self):
        n_steps = 100
        f_vals = np.linspace(0.0, 1.0, n_steps)
        Fs = jax.vmap(
            lambda fii: np.array(
                [[fii, 0.0, 0.0],
                 [0.0, fii, 0.0],
                 [0.0, 0.0, 1.0]]
            )
        )(f_vals)

        energies = np.zeros((4,n_steps))
        Jvals = np.zeros(n_steps)
        for n, F in enumerate(Fs):
            dispGrad = F - np.eye(3)
            energies = energies.at[0,n].set(energy_neo_hookean_rokos_volumetric(dispGrad, self.kappa, self.mu))
            energies = energies.at[1,n].set(energy_neo_hookean_rokos_isochoric(dispGrad, self.kappa, self.mu))
            energies = energies.at[2,n].set(energy_stable_neo_hookean(dispGrad, self.kappa, self.mu))
            energies = energies.at[3,n].set(energy_teran_invertible(dispGrad, self.kappa, self.mu))
            Jvals = Jvals.at[n].set(np.linalg.det(F))

        if plotting:
            plt.figure(1)
            fig, ax1 = plt.subplots()

            # energy
            ax1.set_xlabel('F_11 = F_22 (F_33 = 1)')
            ax1.set_ylabel('Energy')
            legends1 = ax1.plot(f_vals, energies[0,:], linestyle='--', color='r')
            legends2 = ax1.plot(f_vals, energies[1,:], linestyle='--', color='b')
            legends3 = ax1.plot(f_vals, energies[2,:], linestyle='--', color='g')
            legends4 = ax1.plot(f_vals, energies[3,:], linestyle='--', color='c')

            ax2 = ax1.twinx()    # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('J')  # we already handled the x-label with ax1
            legends5 = ax2.plot(f_vals, Jvals, linestyle='--', color='k')

            ax1.legend(legends1+legends2+legends3+legends4+legends5,[
                "Rokos Neo Hookean Volumetric term", 
                "Rokos Neo Hookean Isochoric term", 
                "Stable Neohookean energy", 
                "Teran invertible Neo Hookean", 
                "Volume change J"
            ],
            loc=0, frameon=True)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

            plt.savefig('energy_vs_compression.png')



if __name__ == "__main__":
    unittest.main()
