from optimism.JaxConfig import *
#from optimism import Objective
#from optimism import ConstrainedObjective
from optimism import AlSolver
from optimism import WarmStart
from optimism import EquationSolver

    
def bound_constrained_solve(boundConstrainedObjective,
                            x0, p,
                            alSettings, subSettings,
                            callback=None,
                            sub_problem_callback=None,
                            useWarmStart=True,
                            updatePrecond=True,
                            sub_problem_solver=EquationSolver.trust_region_minimize):

    boundConstrainedObjective.reset_kappa()
    
    xBar0 = boundConstrainedObjective.scaling * x0
    
    if useWarmStart:
        if updatePrecond:
            boundConstrainedObjective.update_precond(xBar0)
        
        dxBar = WarmStart.warm_start_increment(boundConstrainedObjective,
                                               xBar0, p)
        xBar0 += dxBar
        boundConstrainedObjective.p = p
    else:
        dxBar = 0.0
        boundConstrainedObjective.p = p

    if sub_problem_callback: sub_problem_callback(xBar0-dxBar,
                                                  boundConstrainedObjective)
        
    if updatePrecond:
        boundConstrainedObjective.update_precond(xBar0)
        
    xBar = AlSolver.augmented_lagrange_solve(boundConstrainedObjective,
                                             xBar0, p,
                                             alSettings, subSettings,
                                             callback=callback,
                                             sub_problem_callback=sub_problem_callback,
                                             useWarmStart=False,
                                             updatePrecond=False,
                                             sub_problem_solver=sub_problem_solver)
    
    return boundConstrainedObjective.invScaling * xBar
