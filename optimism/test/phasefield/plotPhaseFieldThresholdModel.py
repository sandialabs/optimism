from matplotlib import pyplot as plt
import unittest

from optimism.JaxConfig import *
from optimism.phasefield import PhaseFieldThreshold as Model
from optimism import TensorMath
from optimism.phasefield.MaterialPointSimulator import MaterialPointSimulator

plotting=True

E = 10.0
nu = 0.25
l = 1.0
maxStrain = 0.05
critStrain = 0.025
psiC = 0.5*E*critStrain**2
G0 = psiC/(3.0/16/l) * 1.1
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'critical energy release rate': G0,
         'critical strain energy density': psiC,
         'regularization length': l,
         'kinematics': 'small deformations'}
strainRate = 1.0e-3


class PhaseFieldThresholdUniaxialFixture(unittest.TestCase):
    def setUp(self):
        materialModel = Model.create_material_model_functions(props)
        self.simulator = MaterialPointSimulator(materialModel, maxStrain, strainRate, steps=20)

        
    def testUniaxial(self):
        output = self.simulator.run()

        if plotting:
            fig, axs = plt.subplots(2,2)
            axs[0,0].plot(output.strainHistory, output.stressHistory, marker='o')
            axs[0,0].set(xlabel='strain', ylabel='stress')
            
            axs[1,0].plot(output.strainHistory, output.phaseHistory, marker='o')
            axs[1,0].set(xlabel='strain', ylabel='phase')

            axs[0,1].plot(output.strainHistory, output.energyHistory, marker='o')
            axs[0,1].plot(output.strainHistory, psiC*np.ones_like(output.strainHistory), '--')
            axs[0,1].set(xlabel='strain', ylabel='strain energy density')
            
            fig.set_size_inches(12.0, 10.0)
            plt.tight_layout()
            plt.savefig('phaseUniaxial.pdf')


if __name__ == '__main__':
    unittest.main()
