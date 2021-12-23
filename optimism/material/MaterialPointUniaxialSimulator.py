from collections import namedtuple
import numpy as onp

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import Objective
from optimism import TensorMath


UniaxialOutput = namedtuple('UniaxialOutput', ['times', 'strainHistory', 'stressHistory', 'cauchyStressHistory', 'energyHistory', 'internalVariableHistory'])


class MaterialPointUniaxialSimulator:
    """Generates the uniaxial response of a given potential density function

    The non-axial strain components are determined by minimizing the potential,
    which coincides with stress-free conditions for every stress component
    besides the uniaxial stress.

    Methods
    -------
    run():
      Launches the simluation with the given parameters
    """

    def __init__(self, materialModel, maxStrain, strainRate, steps=10):
        """Constructor

        Args
        ----
          energy_density: function
            Potential density function defining material model
            must have signature  W = energy_density(dispGrad, internalVariables, doUpdate=True)
          maxStrain: float
            Maximum engineering strain for uniaxial test
          strainRate: float
            Strain rate
          update: function (or None)
            Function that determines new internal variable values. May be 'None' if
            the material is elastic
            must have signature  internalVariablesNew = update(dispGrad, internalVariables)
          internalVariables: ndarray (or None)
            Array of initial values of the internal variables
          steps: int
            Number of time steps to take

        Returns
        -------
          Named tuple type UniaxialOutput
          Note stressHistory attribute are Piola stresses
        
        """
        self.strainHistory, self.strainInc = np.linspace(0.0, maxStrain, num=steps, retstep=True)
        self.steps = steps
        self.dt = self.strainInc/strainRate
        self.times = self.strainHistory/strainRate
        self.energy_density = materialModel.compute_energy_density
        self.converged_energy_density_and_stress = jit(value_and_grad(materialModel.compute_output_energy_density))
        self.update = jit(materialModel.compute_state_new)
        self.compute_initial_state = materialModel.compute_initial_state
    
    def run(self):
        def obj_func(freeStrains, p):
            strain = self.makeStrainTensor(freeStrains, p)
            return self.energy_density(strain, p[1])

        solverSettings = EqSolver.get_settings()
        internalVariables =  self.compute_initial_state()
        freeStrains = np.zeros(2)
        
        p = Objective.Params(self.strainHistory[0], internalVariables)
        o = Objective.Objective(obj_func, freeStrains, p)
        
        stressHistory = []
        cauchyStressHistory = []
        energyHistory = []
        internalVariableHistory = []
        for i in range(self.steps):
            p = Objective.param_index_update(p, 0, self.strainHistory[i])

            freeStrains = EqSolver.nonlinear_equation_solve(o, freeStrains, p, solverSettings, useWarmStart=True)
            strain = self.makeStrainTensor(freeStrains, p)
            internalVariables = self.update(strain, internalVariables)
            energyDensity,stress = self.converged_energy_density_and_stress(strain, internalVariables)
            p = Objective.param_index_update(p, 1, internalVariables)
            o.p = p

            F = np.identity(3) + strain
            J = np.linalg.det(F)
            cauchy = stress@F.T/J
            
            stressHistory.append(stress[0,0])
            cauchyStressHistory.append(cauchy[0,0])
            energyHistory.append(energyDensity)
            internalVariableHistory.append(internalVariables)
            
        return UniaxialOutput(onp.array(self.times), onp.array(self.strainHistory),
                              onp.array(stressHistory), onp.array(cauchyStressHistory),
                              onp.array(energyHistory), onp.array(internalVariableHistory))

    
    @staticmethod
    def makeStrainTensor(freeStrains, p):
        uniaxialStrain = p[0]
        return np.diag(np.hstack((uniaxialStrain, freeStrains)))
