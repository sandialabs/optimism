from matplotlib import pyplot as plt
import unittest

from optimism.JaxConfig import *
from optimism.phasefield import SandiaModel as Model
from optimism import TensorMath
from optimism.phasefield.MaterialPointSimulator import MaterialPointSimulator

plotting=True

E = 10.0
nu = 0.25
l = 1.0
H = 1e-1 * E
Y0 = 0.01*E
maxStrain = 0.05
critStrain = 0.025
psiC = 0.5*E*((H*critStrain + Y0)/(H + E))**2
G0 = psiC/(3.0/16/l) * 1.1
void0 = 0.0
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'yield strength': Y0,
         'hardening model': 'linear',
         'hardening modulus': H,
         'critical energy release rate': G0,
         'critical strain energy density': psiC,
         'regularization length': l,
         'kinematics': 'small deformations',
         'void growth prefactor': 1.0,
         'void growth exponent': 1.0,
         'initial void fraction': 0.0}
strainRate = 1.0e-3


class PhaseFieldUniaxialFixture(unittest.TestCase):
    def setUp(self):
        materialModel = Model.create_material_model_functions(props)
        self.simulator = MaterialPointSimulator(materialModel, maxStrain, strainRate, steps=20)

        
    def testUniaxial(self):
        output = self.simulator.run()
        
        eqpsHistory = vmap(lambda Q : Q[Model.STATE_EQPS])(output.internalVariableHistory)
        voidHistory = vmap(lambda Q : Q[Model.STATE_VOID_FRACTION])(output.internalVariableHistory)
        psiCHistory = Model.update_critical_energy(psiC, voidHistory)

        if plotting:
            fig, axs = plt.subplots(2,3)
            axs[0,0].plot(output.strainHistory, output.stressHistory, marker='o')
            axs[0,0].set(xlabel='strain', ylabel='stress')
            
            axs[1,0].plot(output.strainHistory, output.phaseHistory, marker='o')
            axs[1,0].set(xlabel='strain', ylabel='phase')
            
            axs[0,1].plot(output.strainHistory, eqpsHistory, marker='o')
            axs[0,1].set(xlabel='strain', ylabel='equivalent plastic strain')
            
#            axs[1,1].plot(output.strainHistory, output.energyHistory, marker='o')
#            axs[1,1].set(xlabel='strain', ylabel='Potential density')
            axs[1,1].plot(output.strainHistory, voidHistory, marker='o')
            axs[1,1].set(xlabel='strain', ylabel='Void fraction')

            axs[0,2].plot(output.strainHistory, psiCHistory, marker='o')
            axs[0,2].plot(output.strainHistory, np.ones_like(psiCHistory)*psiC, linestyle='dashed')
            axs[0,2].set(xlabel='strain', ylabel=r'$\Psi$')
            axs[0,2].legend((r'$\Psi$',r'$\Psi_c$',r'$\Psi_{c0}$'))
            
            fig.set_size_inches(12.0, 10.0)
            plt.tight_layout()
            plt.savefig('phaseUniaxial.pdf')


if __name__ == '__main__':
    unittest.main()
