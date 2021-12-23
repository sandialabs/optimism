from collections import namedtuple
import numpy as onp
from jax import disable_jit

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import AlSolver
from optimism import Objective
from optimism import ConstrainedObjective
from optimism import TensorMath


UniaxialOutput = namedtuple('UniaxialOutput',
                            ['times', 'strainHistory', 'stressHistory',
                             'kirchhoffStressHistory', 'energyHistory',
                             'internalVariableHistory', 'phaseHistory'])

FREE_STRAINS = slice(0,2)
PHASE = 2
NUM_UNKNOWNS = 3


class MaterialPointSimulator:
    """Generates the uniaxial response of a given phase field potential density function

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
          materialModel: namedtuple (of functions)function
            Set of functions for desired phase field fracture material model
          maxStrain: float
            Maximum engineering strain for uniaxial test
          strainRate: float
            Strain rate
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
        self.energy_density_and_fluxes = jit(value_and_grad(materialModel.compute_output_energy_density, (0,1)))
        self.state_new = jit(materialModel.compute_state_new)
        self.compute_initial_state = materialModel.compute_initial_state

    
    def run(self):
        gradPhase = np.zeros(3)
        def obj_func(unknowns, p):
            strain, phase = self.makeKinematics(unknowns, p)
            internalVariables = p[1]
            return self.energy_density(strain, phase, gradPhase, internalVariables)

        def constraint_func(unknowns, p):
            return np.array([unknowns[PHASE]])

        alSettings = AlSolver.get_settings(target_constraint_decrease_factor=0.9,
                                           num_initial_low_order_iterations=2)
        trSettings = EqSolver.get_settings()
        internalVariables = self.compute_initial_state()
        
        
        p = Objective.Params(self.strainHistory[0], internalVariables)

        unknowns = np.zeros(NUM_UNKNOWNS)
        strain, phase = self.makeKinematics(unknowns, p)
        #print("uncon hessian = ", jit(hessian(obj_func))(unknowns, p))
        energyDensity, fluxes = self.energy_density_and_fluxes(strain, phase, gradPhase, internalVariables)
        stress, phaseForce = fluxes
        lam0 = np.array([phaseForce])
        kappa0 = np.array([10.0*phaseForce]) # ??
        o = ConstrainedObjective.ConstrainedObjective(obj_func,
                                                      constraint_func,
                                                      unknowns,
                                                      p,
                                                      lam0,
                                                      kappa0)
        
        strainEnergyDensityHistory = []
        stressHistory = []
        kirchhoffStressHistory = []
        energyHistory = []
        phaseHistory  = []
        internalVariableHistory = []

        for i in range(self.steps):
            print('-----------------------------------')
            print('STEP ', i)
            p = Objective.param_index_update(p, 0, self.strainHistory[i])
            unknowns = AlSolver.augmented_lagrange_solve(o, unknowns, p, alSettings, trSettings)
            
            strain, phase = self.makeKinematics(unknowns, p)
            internalVariables = self.state_new(strain, phase, gradPhase, internalVariables)
            p = Objective.param_index_update(p, 1, internalVariables)

            # compute outputs
            energyDensity, fluxes = self.energy_density_and_fluxes(strain, phase, gradPhase, internalVariables)
            stress, phaseForce = fluxes
            F = np.identity(3) + strain
            stressK = stress@F.T
            stressHistory.append(stress[0,0])
            kirchhoffStressHistory.append(stressK[0,0])
            energyHistory.append(energyDensity)
            internalVariableHistory.append(internalVariables)
            phaseHistory.append(phase)
            
        return UniaxialOutput(times=onp.array(self.times),
                              strainHistory=onp.array(self.strainHistory),
                              stressHistory=onp.array(stressHistory),
                              kirchhoffStressHistory=onp.array(kirchhoffStressHistory),
                              energyHistory=onp.array(energyHistory),
                              internalVariableHistory=onp.array(internalVariableHistory),
                              phaseHistory=onp.array(phaseHistory))

    
    @staticmethod
    def makeKinematics(unknowns, p):
        uniaxialStrain = p[0]
        freeStrains = unknowns[FREE_STRAINS]
        strain = np.diag( np.hstack((uniaxialStrain,freeStrains)) )
        phase = unknowns[PHASE]
        return strain, phase
