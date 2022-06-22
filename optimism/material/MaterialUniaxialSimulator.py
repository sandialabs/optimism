from collections import namedtuple
import numpy as onp

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import Objective
from optimism import TensorMath


UniaxialOutput = namedtuple('UniaxialOutput', ['time', 'strainHistory', 'stressHistory',
                                               'energyHistory', 'internalVariableHistory'])


class MaterialUniaxialSimulator:
    """Generates the uniaxial response of a given potential density function

    The non-axial strain components are determined by minimizing the potential,
    which coincides with stress-free conditions for every stress component
    besides the uniaxial stress.

    Methods
    -------
    run():
      Launches the simluation with the given parameters
    """

    def __init__(self, materialModel, strain_history, maxTime, steps=10):
        """Constructor

        Args
        ----
        materialModel: MaterialModel object
            Material to subject to uniaxial stress test
        strain_history: callable (float) -> float
            The tensile strain as a function of time
        maxTime: float
            Upper limit of the time interval
        steps: int
            Number of time steps to take
        """
        self.steps = steps
        self.times = np.linspace(0.0, maxTime, num=steps)
        self.uniaxialStrainHistory = strain_history(self.times)
        self.energy_density = materialModel.compute_energy_density
        self.converged_energy_density_and_stress = jit(value_and_grad(materialModel.compute_output_energy_density))
        self.update = jit(materialModel.compute_state_new)
        self.compute_initial_state = materialModel.compute_initial_state
    
    def run(self):
        """ Execute the uniaxial test
        Returns
        -------
        UniaxialOutput: named tuple of constitutive output
        """
        
        def obj_func(freeStrains, p):
            strain = self.makeStrainTensor(freeStrains, p)
            return self.energy_density(strain, p[1])

        solverSettings = EqSolver.get_settings()
        internalVariables =  self.compute_initial_state()
        freeStrains = np.zeros(2)
        
        p = Objective.Params(self.uniaxialStrainHistory[0], internalVariables)
        o = Objective.Objective(obj_func, freeStrains, p)

        strainHistory = []
        stressHistory = []
        kirchhoffStressHistory = []
        energyHistory = []
        internalVariableHistory = []
        for i in range(self.steps):
            p = Objective.param_index_update(p, 0, self.uniaxialStrainHistory[i])

            freeStrains = EqSolver.nonlinear_equation_solve(o, freeStrains, p, solverSettings, useWarmStart=True)
            strain = self.makeStrainTensor(freeStrains, p)
            internalVariables = self.update(strain, internalVariables)
            energyDensity,stress = self.converged_energy_density_and_stress(strain, internalVariables)
            p = Objective.param_index_update(p, 1, internalVariables)
            o.p = p

            F = np.identity(3) + strain
            J = np.linalg.det(F)
            cauchy = stress@F.T/J

            strainHistory.append(strain)
            stressHistory.append(stress)
            energyHistory.append(energyDensity)
            internalVariableHistory.append(internalVariables)
            
        return UniaxialOutput(onp.array(self.times), onp.array(strainHistory),
                              onp.array(stressHistory), onp.array(energyHistory),
                              onp.array(internalVariableHistory))

    
    @staticmethod
    def makeStrainTensor(freeStrains, p):
        uniaxialStrain = p[0]
        return np.diag(np.hstack((uniaxialStrain, freeStrains)))
