from collections import namedtuple
import numpy as onp

import jax
import jax.numpy as np

from . import EquationSolver_Immersed_2 as EqSolver
from optimism import Objective
from optimism import TensorMath


UniaxialOutput = namedtuple('UniaxialOutput', ['time', 'strainHistory', 'stressHistory',
                                               'energyHistory', 'internalVariableHistory'])
UniaxialOutput.__doc__ = """\
Output from a uniaxial tension test on a material.

Attributes
----------
time: array
    discrete time points in the unixial data
strainHistory: array
    displacement gradient tensor at each time point
stressHistory: array
    Piola stress tensor at each time point
energyHistory: array
    Potential function value at each time point
internalVariableHistory: array
    Collection of internal variables for the given material at each time point.
"""


def run(materialModel, strain_history, maxTime, steps=10, tol=1e-3):
    """ Generates the uniaxial response of a given material
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
    
    Returns
    -------
    UniaxialOutput: named tuple of constitutive output
    """
    timePoints = np.linspace(0.0, maxTime, num=steps)
    dt = timePoints[1] - timePoints[0]
    uniaxialStrainHistory = strain_history(timePoints)
    energy_density = materialModel.compute_energy_density
    converged_energy_density_and_stress = jax.jit(jax.value_and_grad(materialModel.compute_energy_density))
    update = jax.jit(materialModel.compute_state_new)
        
    def obj_func(freeStrains, p):
        strain = makeStrainTensor_(freeStrains, p)
        return energy_density(strain, p[1], dt)

    uniaxialTolerance=tol
    solverSettings = EqSolver.get_settings(tol=uniaxialTolerance)
    internalVariables =  materialModel.compute_initial_state()
    freeStrains = np.zeros(2)
        
    p = Objective.Params(uniaxialStrainHistory[0], internalVariables)
    o = Objective.Objective(obj_func, freeStrains, p)

    strainHistory = []
    stressHistory = []
    energyHistory = []
    internalVariableHistory = []
    for i in range(steps):
        print('---------------------------')
        print(f'Step {i}')
        p = Objective.param_index_update(p, 0, uniaxialStrainHistory[i])

        freeStrains, solverSuccess = EqSolver.nonlinear_equation_solve(o, freeStrains, p, solverSettings, useWarmStart=True)
        strain = makeStrainTensor_(freeStrains, p)
        internalVariables = update(strain, internalVariables, dt)
        energyDensity,stress = converged_energy_density_and_stress(strain, internalVariables, dt)
        p = Objective.param_index_update(p, 1, internalVariables)
        o.p = p

        strainHistory.append(strain)
        stressHistory.append(stress)
        energyHistory.append(energyDensity)
        internalVariableHistory.append(internalVariables)
            
    return UniaxialOutput(onp.array(timePoints), onp.array(strainHistory),
                          onp.array(stressHistory), onp.array(energyHistory),
                          onp.array(internalVariableHistory))

    
def makeStrainTensor_(freeStrains, p):
    uniaxialStrain = p[0]
    return np.diag(np.hstack((uniaxialStrain, freeStrains)))
