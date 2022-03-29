from optimism.JaxConfig import *
import optimism.EquationSolver as EqSolver
from optimism.NewtonSolver import globalized_newton_step
from optimism.NewtonSolver import newton_step
from optimism import WarmStart

Settings = namedtuple('Settings',
                      ['penalty_scaling',
                       'target_constraint_decrease_factor',
                       'relative_gmres_tol',
                       'max_gmres_iters',
                       'use_second_order_update',
                       'use_newton_only',
                       'num_initial_low_order_iterations',
                       'inverse_ncp_hessian_bound',
                       'max_al_iters',
                       'tol'])

def get_settings(penalty_scaling=4.,
                 target_constraint_decrease_factor=0.75,
                 relative_gmres_tol=2e-2,
                 max_gmres_iters=100,
                 use_second_order_update=True,
                 use_newton_only=False,
                 num_initial_low_order_iterations=3,
                 inverse_ncp_hessian_bound=1e-2,
                 max_al_iters=100,
                 tol=1e-8):
    
    return Settings(penalty_scaling,
                    target_constraint_decrease_factor,
                    relative_gmres_tol,
                    max_gmres_iters,
                    use_second_order_update,
                    use_newton_only,
                    num_initial_low_order_iterations,
                    inverse_ncp_hessian_bound,
                    max_al_iters,
                    tol)

norm = np.linalg.norm


def solve_sub_step(alObjective, x, ncpErrorOld, alSettings, subSettings, sub_problem_solver, sub_problem_callback=None):
    x = sub_problem_solver(alObjective, x, subSettings, sub_problem_callback)
    
    c = alObjective.constraint(x)
    kappa = alObjective.kappa
    alObjective.lam = np.maximum(alObjective.lam-kappa*c, 0.0)
    
    ncpError = np.abs( alObjective.ncp(x) )
    poorProgress = ncpError > alSettings.target_constraint_decrease_factor * ncpErrorOld

    if (np.any(poorProgress)):
        print('Poor progress on ncp detected, increasing some penalty parameters')
        alObjective.kappa = kappa.at[poorProgress].set(alSettings.penalty_scaling*kappa[poorProgress])
        
    return x, ncpError


def linear_update(alObjective, x, rhs_func, alSettings):
    xl = np.hstack((x, alObjective.lam))
    xsize = x.size

    
    def residual_jac_vec(vxl):
        return alObjective.constrained_jacobian_vec(xl, vxl)

    
    def precond(vl):
        xR = vl[:xsize]
        lR = vl[xsize:]
        D = alObjective.ncp_hessian(x)

        constraintDiffCap = alSettings.inverse_ncp_hessian_bound
        nearSingularValues = D > -constraintDiffCap
        
        DInv = 1.0 / D.at[nearSingularValues].set(-constraintDiffCap)
        DInvCapped = D.at[nearSingularValues].set(0.0)
        
        deltaL = lR * DInv
        deltaX = alObjective.apply_precond(xR - alObjective.jacobian_l_vec(x, lR * DInvCapped))
        return np.hstack((deltaX, deltaL))
    
    dxl, solveSuccess = newton_step(rhs_func, residual_jac_vec,
                                    xl, settings=alSettings, precond=precond)
            
    return dxl[:xsize], dxl[xsize:], solveSuccess
        

def augmented_lagrange_solve(alObjective, x, p, alSettings, subSettings,
                             callback=None,
                             sub_problem_callback=None,
                             sub_problem_solver=EqSolver.trust_region_minimize,
                             useWarmStart=True,
                             updatePrecond=True,
                             updatePrecondBeforeWarmStart=True):

    if useWarmStart:
        if updatePrecondBeforeWarmStart:
            alObjective.update_precond(x)
        x += WarmStart.warm_start_increment(alObjective, x, p)
        alObjective.p = p
    else:
        alObjective.p = p

    if updatePrecond:
        alObjective.update_precond(x)
        
    hugeVal = 1e64
    ncpError = hugeVal * np.ones(alObjective.lam.shape)
    errorNorm = hugeVal
    
    maxAlIters = alSettings.max_al_iters

    initialTolScaling = 100.0
    tolRampDownIters=3
    
    for it in range(maxAlIters):

        if it < tolRampDownIters:
            tolScaling = np.power(initialTolScaling, (tolRampDownIters-it) / tolRampDownIters)
            settings = EqSolver.settings_with_new_tol(subSettings,
                                                      tolScaling * subSettings.tol)
        else:
            settings = subSettings
            
        if callback: callback(x, alObjective.p)
        
        updatePrecond=False
        if (alSettings.use_second_order_update and it >=alSettings.num_initial_low_order_iterations) or alSettings.use_newton_only:
            
            dx, dl, solveSuccess = linear_update(alObjective,
                                                 x,
                                                 alObjective.constrained_residual,
                                                 alSettings)

            if solveSuccess!=0:
                updatePrecond=True

            lamSave = np.array(alObjective.lam)
            
            maxLinesearchIters=10
            for linesearch in range(maxLinesearchIters):
                if linesearch==maxLinesearchIters-1:
                    updatePrecond=True
                
                y = x + dx
                alObjective.lam = alObjective.lam + dl

                trialErrorNorm = norm(alObjective.total_residual(y))
                
                if norm(trialErrorNorm  < errorNorm):
                    # print('Force error after 2nd order update success = ', norm(alObjective.gradient(y)))
                    print('Total error after 2nd order update = ', trialErrorNorm)
                    errorNorm = trialErrorNorm
                    x = y
                    break
                else:
                    print('Total error after 2nd order update = ', trialErrorNorm,
                          ', no improvement')
                    alObjective.lam = lamSave
                    dx *= 0.2
                    dl *= 0.2

        if updatePrecond:
            #print('GMRES solver was struggling.')
            alObjective.update_precond(x)

        if not alSettings.use_newton_only:
            x, ncpError = solve_sub_step(alObjective,
                                         x,
                                         ncpError,
                                         alSettings,
                                         settings,
                                         sub_problem_solver,
                                         sub_problem_callback)
            
            forceErrorNorm = norm(alObjective.gradient(x))
            print('force error = ', forceErrorNorm)
            print('ncp error = ', norm(ncpError))
            errorNorm = norm(alObjective.total_residual(x))
            print('total error = ', errorNorm)
            
            if errorNorm < alSettings.tol:
                if callback: callback(x, alObjective.p)
                return x

    raise NameError('Loadstep failed to converge in', maxAlIters, 'iterations.')
        
    return x

