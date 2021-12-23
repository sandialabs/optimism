from matplotlib import pyplot as plt
import unittest
from optimism.JaxConfig import *
from optimism.phasefield import PhaseFieldThresholdPlastic
from optimism import TensorMath
from optimism.phasefield.MaterialPointSimulator import MaterialPointSimulator

plotting=False

E = 10.0
nu = 0.25
# Gc goes here in order
l = 1.0
H = 1e-1 * E
Y0 = 0.01*E

maxStrain = 0.05
critStrain = 0.025
psiC = 0.5*E*((H*critStrain + Y0)/(H + E))**2
Gc = psiC/(3.0/16/l)

props = PhaseFieldThresholdPlastic.make_properties(E, nu, Gc, l, Y0, H)

def energy_density(dispGrad, phase, phaseGrad, internalVariables, doUpdate=True):
    return PhaseFieldThresholdPlastic.energy_density(dispGrad, phase, phaseGrad,
                                                     internalVariables, props=props, doUpdate=doUpdate)

def update(dispGrad, phase, phaseGrad, internalVariables):
    return PhaseFieldThresholdPlastic.compute_state_new(dispGrad, phase, phaseGrad,
                                                        internalVariables, props=props)


class PhaseFieldUniaxialFixture(unittest.TestCase):

    @unittest.skip
    def testUniaxial(self):

        internalVariables = PhaseFieldThresholdPlastic.make_initial_state()
        
        strainRate = 1.0e-3
        simulator = MaterialPointSimulator(energy_density, energy_density, update, maxStrain, strainRate, internalVariables, steps=20)
        output = simulator.run()
        
        eqpsHistory = [Q[PhaseFieldThresholdPlastic.STATE_EQPS] for Q in output.internalVariableHistory]

        if plotting:
            fig, axs = plt.subplots(2,2)
            axs[0,0].plot(output.strainHistory, output.stressHistory, marker='o')
            axs[0,0].set(xlabel='strain', ylabel='stress')
            
            axs[1,0].plot(output.strainHistory, output.phaseHistory, marker='o')
            axs[1,0].set(xlabel='strain', ylabel='phase')
            
            axs[0,1].plot(output.strainHistory, eqpsHistory, marker='o')
            axs[0,1].set(xlabel='strain', ylabel='equivalent plastic strain')
            
            axs[1,1].plot(output.strainHistory, output.energyHistory, marker='o')
            axs[1,1].set(xlabel='strain', ylabel='Potential density')
            
            fig.set_size_inches(12.0, 10.0)
            plt.tight_layout()
            plt.savefig('phaseUniaxial.pdf')


if __name__ == '__main__':
    unittest.main()
