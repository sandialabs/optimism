from optimism.JaxConfig import *
from jax import custom_jvp, custom_vjp

from optimism import EquationSolver
from optimism import Objective
from optimism import WarmStart

@partial(custom_vjp, nondiff_argnums=(0,1))
def nonlinear_solve(mechanicalEnergy, settings, UuGuess, designParams):
    p = Objective.param_index_update(mechanicalEnergy.p, 2, designParams)
    return EquationSolver.nonlinear_equation_solve(mechanicalEnergy,
                                                   UuGuess, p, settings)


def nonlinear_solve_f(mechanicalEnergy, settings, UuGuess, designParams):
    Uu = nonlinear_solve(mechanicalEnergy, settings, UuGuess, designParams)
    return Uu, (Uu, designParams)


def nonlinear_solve_b(mechanicalEnergy, settings, rdata, v):
    Uu,designParams = rdata
    mechanicalEnergy.p = Objective.param_index_update(mechanicalEnergy.p, 2, designParams)
    
    hess_vec_func = lambda w: mechanicalEnergy.hessian_vec(Uu, w)
    
    results = EquationSolver.solve_trust_region_minimization(0.0*Uu,
                                                             v,
                                                             hess_vec_func,
                                                             mechanicalEnergy.apply_precond,
                                                             lambda x: x,
                                                             np.inf,
                                                             settings)
    
    lam = results[0]
    return (np.zeros_like(Uu)*v[0], mechanicalEnergy.vec_jacobian_p2(Uu, lam)[0])

nonlinear_solve.defvjp(nonlinear_solve_f, nonlinear_solve_b) 


### new version

@partial(custom_vjp, nondiff_argnums=(0,1))
def nonlinear_solve_with_state(mechanicalEnergy, settings, UuGuess, p):

    mechanicalEnergy.update_precond(UuGuess)
    UuGuess += WarmStart.warm_start_increment_jax_safe(mechanicalEnergy, UuGuess, p[0])

    Uu = EquationSolver.nonlinear_equation_solve(mechanicalEnergy,
                                                 UuGuess,
                                                 p,
                                                 settings,
                                                 useWarmStart=False)
    return Uu


def nonlinear_solve_with_state_f(mechanicalEnergy, settings, UuGuess, p):
    Uu = nonlinear_solve_with_state(mechanicalEnergy, settings, UuGuess, p)
    return Uu, (Uu, p)


def nonlinear_solve_with_state_b(mechanicalEnergy, settings, rdata, v):
    Uu, p = rdata

    mechanicalEnergy.p = p
    hess_vec_func = lambda w: mechanicalEnergy.hessian_vec(Uu, w)

    UuZeros = np.zeros_like(Uu)
    results = EquationSolver.solve_trust_region_minimization(UuZeros,
                                                             v,
                                                             hess_vec_func,
                                                             mechanicalEnergy.apply_precond,
                                                             lambda x: x,
                                                             np.inf,
                                                             settings)
    lam = results[0]

    if p[0] != None:
        dp0 = mechanicalEnergy.vec_jacobian_p0(Uu, lam)[0]
    else:
        dp0 = None

    if p[1] != None:
        dp1 = mechanicalEnergy.vec_jacobian_p1(Uu, lam)[0]
    else:
        dp1 = None

    if p[2] != None:
        dp2 = mechanicalEnergy.vec_jacobian_p2(Uu, lam)[0]
    else:
        dp2 = None

    if p[4] != None:
        dp4 = mechanicalEnergy.vec_jacobian_p4(Uu, lam)[0]
    else:
        dp4 = None
    
    return (UuZeros,
            Objective.Params(dp0, dp1, dp2, None, dp4))


nonlinear_solve_with_state.defvjp(nonlinear_solve_with_state_f, nonlinear_solve_with_state_b) 


