from scipy.sparse import save_npz
from scipy import optimize
from collections import deque

from optimism.JaxConfig import *
# from optimism import ScalarRootFind
# from optimism.Timer import Timer
from optimism import WarmStart


Settings = namedtuple('Settings', ['t1','t2','eta1','eta2','eta3','max_trust_iters','tol','max_spg_iters','max_cumulative_spg_iters','spg_tol','spg_inexact_solve_ratio','tr_size','min_tr_size','check_stability','use_preconditioned_inner_product_for_spg','spg_use_nonmonotone','spg_nonmonotone_iter_limit_to_enforce_decrease','use_incremental_objective','cauchy_point_sufficient_decrease_factor','cauchy_point_decrease_tol','cauchy_point_max_line_search_iters','min_spectral_step_length','max_spectral_step_length','debug_info'])


def get_settings(t1=0.25, t2=1.75, eta1=1e-10, eta2=0.1, eta3=0.5,
                 max_trust_iters=100,
                 tol=1e-8,
                 max_spg_iters=25,
                 max_cumulative_spg_iters=1000,
                 spg_tol=None,
                 spg_inexact_solve_ratio=1e-4,
                 tr_size=2.0,
                 min_tr_size=1e-8,
                 check_stability=False,
                 use_preconditioned_inner_product_for_spg=False,
                 spg_use_nonmonotone=True,
                 spg_nonmonotone_iter_limit_to_enforce_decrease=10,
                 use_incremental_objective=False,
                 cauchy_point_sufficient_decrease_factor=1e-4,
                 cauchy_point_decrease_tol=1e-8,
                 cauchy_point_max_line_search_iters=25,
                 min_spectral_step_length=1e-12,
                 max_spectral_step_length=1e12,
                 debug_info=True):
    """Must have spg_tol < tol to avoid inexactness of subproblem
    solves preventing convergence of trust region algorithm.
    """
    if spg_tol==None:
        spg_tol = 0.2*tol

    return Settings(t1, t2, eta1, eta2, eta3,
                    max_trust_iters=max_trust_iters,
                    tol=tol,
                    max_spg_iters=max_spg_iters,
                    max_cumulative_spg_iters=max_cumulative_spg_iters,
                    spg_tol=spg_tol,
                    spg_inexact_solve_ratio=spg_inexact_solve_ratio,
                    tr_size=tr_size,
                    min_tr_size=min_tr_size,
                    check_stability=check_stability,
                    use_preconditioned_inner_product_for_spg=use_preconditioned_inner_product_for_spg,
                    spg_use_nonmonotone=spg_use_nonmonotone,
                    spg_nonmonotone_iter_limit_to_enforce_decrease=spg_nonmonotone_iter_limit_to_enforce_decrease,
                    use_incremental_objective=use_incremental_objective,
                    cauchy_point_sufficient_decrease_factor=cauchy_point_sufficient_decrease_factor,
                    cauchy_point_decrease_tol=cauchy_point_decrease_tol,
                    cauchy_point_max_line_search_iters=cauchy_point_max_line_search_iters,
                    min_spectral_step_length=min_spectral_step_length,
                    max_spectral_step_length=max_spectral_step_length,
                    debug_info=debug_info)


def settings_with_new_tol(oldSettings, newTol):
    newSettings = Settings(t1=oldSettings.t1,
                           t2=oldSettings.t2,
                           eta1=oldSettings.eta1,
                           eta2=oldSettings.eta2,
                           eta3=oldSettings.eta3,
                           max_trust_iters=oldSettings.max_trust_iters,
                           tol=newTol, # here
                           max_spg_iters=oldSettings.max_spg_iters,
                           max_cumulative_spg_iters=oldSettings.max_cumulative_spg_iters,
                           spg_tol=0.2*newTol, # here
                           spg_inexact_solve_ratio=oldSettings.spg_inexact_solve_ratio,
                           tr_size=oldSettings.tr_size,
                           min_tr_size=oldSettings.min_tr_size,
                           check_stability=oldSettings.check_stability,
                           use_preconditioned_inner_product_for_spg=oldSettings.use_preconditioned_inner_product_for_spg,
                           spg_use_nonmonotone=oldSettings.spg_use_nonmonotone,
                           spg_nonmonotone_iter_limit_to_enforce_decrease=oldSettings.spg_nonmonotone_iter_limit_to_enforce_decrease,
                           use_incremental_objective=oldSettings.use_incremental_objective,
                           cauchy_point_sufficient_decrease_factor=oldSettings.cauchy_point_sufficient_decrease_factor,
                           cauchy_point_decrease_tol=oldSettings.cauchy_point_decrease_tol,
                           cauchy_point_max_line_search_iters=oldSettings.cauchy_point_max_line_search_iters,
                           min_spectral_step_length=oldSettings.min_spectral_step_length,
                           max_spectral_step_length=oldSettings.max_spectral_step_length,
                           debug_info=oldSettings.debug_info)
                           

    return newSettings


boundaryString='boundary'
interiorString='interior'
cauchyString='cauchy pt'


def is_on_boundary(stepType):
    return stepType==boundaryString


def subproblem_optimality(xNew, x, d, bounds, trSize):
    chi = project_onto_tr(xNew - d, x, bounds, trSize) - xNew
    return chi@chi


def kouri_exact_line_search(ds, sBs, q, qMax, settings):
    alpha = -ds/sBs
    return alpha


def nonmonotone_line_search(ds, sBs, q, qMax, settings):
    gamma = 1e-4 # Armijo sufficient decrease parameter
    b = (1 - gamma)*ds
    alpha = (-b + np.sqrt(b**2 - 2*sBs*(q - qMax))) / sBs
    return alpha


def solve_spg_subproblem(x, cauchyStep, r, bounds, hess_vec_func, precond, trSize, settings):
    # minimize r@z + 0.5*z@B@z st y in C, ||y|| <= delta
    z = cauchyStep
    xNew = x + z
    Bz = hess_vec_func(z)
    d = r + Bz
    q = r@z + 0.5*z@Bz
    chi2 = subproblem_optimality(xNew, x, d, bounds, trSize)

    # print('norm g', np.linalg.norm(r))

    spgTolSquared = max(settings.spg_tol**2,
                        chi2*settings.spg_inexact_solve_ratio**2)
    if chi2 < spgTolSquared:
        return z, q, np.sqrt(chi2), cauchyString, 0

    lamMin = settings.min_spectral_step_length
    lamMax = settings.max_spectral_step_length
    lam = max(lamMin, min(1/np.linalg.norm(d), lamMax))
    # print('initial lam', lam)

    line_search = nonmonotone_line_search if settings.spg_use_nonmonotone else kouri_exact_line_search

    # set up nonmonotone descent check data
    M = settings.spg_nonmonotone_iter_limit_to_enforce_decrease
    qHistory = deque([-np.inf for i in range(M-1)])
    qHistory.append(float(q))

    # SPG iterations
    
    for i in range(settings.max_spg_iters):
        # print('\t----------')
        # print('\tspg iteration ', i)
        s = project_onto_tr(xNew - lam*d, x, bounds, trSize) - xNew
        # print('\tnorm of tentative subproblem solution', np.linalg.norm(z + s))
        Bs = hess_vec_func(s)
        sBs = s@Bs
        ds = d@s
        qMax = max(qHistory)
        alpha = line_search(ds, sBs, q, qMax, settings)
        alpha = min(1.0, alpha) if sBs > 0 else 1.0

        z += alpha*s
        d += alpha*Bs
        q += alpha*(ds + 0.5*alpha*sBs)
        xNew = x + z

        chi2 = subproblem_optimality(xNew, x, d, bounds, trSize)

        # print('\txNew = ', xNew)
        # print('\tq = ', q)
        # print('\tlam = ', lam)
        # print('\toptimality', np.sqrt(chi2))
        # print('\tstep length alpha', alpha)
        # print('\tsubproblem step length', np.linalg.norm(z))

        qHistory.popleft()
        qHistory.append(float(q))
        
        if chi2 < spgTolSquared:
            return z, q, np.sqrt(chi2), boundaryString, i+1

        lam = max(lamMin, min(lamMax, s@s/sBs)) if sBs > 0 else lamMax
            
    return z, q, np.sqrt(chi2), interiorString+'_', i+1

    
def print_min_banner(objective, modelObjective, res, modelRes, spgIters, trSize, onBoundary, willAccept, settings):
    if settings.debug_info==False: return
    
    willAccept = " True" if willAccept else "False"
    
    print('obj=', f"{objective:15.9}",
          ', model obj=', f"{modelObjective:15.9}",
          ', res=', f"{res:15.9}", ', model res=', f"{modelRes:15.9}",
          ', spg iters=', f"{spgIters:3}",
          ', tr size=', f"{trSize:12.6}",
          ', ', f"{onBoundary:9}",
          ', accepted=', willAccept)


# utility function to save a matrix for later analysis offline
def output_matrix(objective, x):
    hess = objective.hessian(x)
    hess = SparseCholesky.convert_to_sparse(hess)
    print('saving matrix')
    save_npz('matrix', hess)
    exit() 


def is_converged(objective, x, realO, modelO, realOptimality,
                 modelOptimality, spgIters, trSize, settings):

    if realOptimality < settings.tol:
        print_min_banner(realO,
                         modelO,
                         realOptimality,
                         modelOptimality,
                         spgIters,
                         trSize,
                         interiorString,
                         True,
                         settings)
        if settings.check_stability:
            objective.check_stability(x)

        print('') # a bit of output formatting
            
        return True
    return False


def project(x, bounds):
    """Projects onto the feasible set. """
    lb = bounds[:,0]
    ub = bounds[:,1]
    x = np.maximum(lb, np.minimum(x, ub))
    return x


def project_onto_tr(x, xk, bounds, trSize):
    """Projects onto intersection of feasible set and trust region boundary.
    """
    d = project(x, bounds) - xk
    dd = d@d
    if dd <= trSize*trSize:
        return project(x, bounds)

    def f(t):
        r = project(xk + t*(x - xk), bounds) - xk
        return r@r - trSize*trSize

    #t = ScalarRootFind.rtsafe(f, x, np.array([0.0, 1.0]), rtsafeSettings)
    t, results = optimize.brentq(f, 0.0, 1.0, full_output=True)
    # print('Brent method iterations', results.iterations)
    #if not results.converged:
    #    raise RuntimeError('TrustRegionSPG: Root finder failed')
    return project(xk + t*(x - xk), bounds)


def find_generalized_cauchy_point(x, g, hess_vec_func, bounds, alpha, trSize, settings):
    mu0 = settings.cauchy_point_sufficient_decrease_factor
    qTol = settings.cauchy_point_decrease_tol
    cutback = 0.2
    maxLineSearchIters = settings.cauchy_point_max_line_search_iters

    def m(s): return 0.5*s@hess_vec_func(s) + g@s
    deltaSquared = trSize*trSize

    # print('CP line seach')
    # print('g = ', g, 'alpha = ', alpha)
    
    s = project(x - alpha*g, bounds) - x
    intialStepAcceptable = m(s) <= mu0*g@s
    if intialStepAcceptable:
        # forward track
        # print('Forward tracking cp')
        q = m(s)
        alphaTry = alpha/cutback
        sTry = project(x - alphaTry*g, bounds) - x
        qTry = m(sTry)
        search = True
        i = 0
        while search:
            # print('i', i)
            if qTry <= mu0*g@sTry and np.abs(q-qTry) > qTol*np.abs(q):
                alpha = alphaTry
                s = sTry
            else:
                search = False
                             
            alphaTry /= cutback
            sTry = project(x - alphaTry*g, bounds) - x
            qTry = m(sTry)
            i += 1

            # We need to check against the trust region size
            # here in case the Hessian has negative eigenvalues.
            if sTry@sTry >= deltaSquared or i == maxLineSearchIters:
                search = False
    else:
        # back track
        # print('Back tracking cp')
        i = 0
        search = True
        while search:
            # print('i', i)
            alpha *= cutback
            s = project(x - alpha*g, bounds) - x
            i += 1
            search = m(s) > mu0*g@s and i < maxLineSearchIters
        if i == maxLineSearchIters:
            raise RuntimeError('No acceptable Cauchy point found after maximum allowed line search iterations')
            
    ss = s@s
    if ss > deltaSquared:
        print('generalized cauchy step outside trust region at radius', np.sqrt(ss))
        i = 0
        search = True
        while search:
            alpha *= cutback
            s = project(x - alpha*g, bounds) - x
            ss = s@s
            i += 1
            search = ss > deltaSquared and i < maxLineSearchIters
        if i == maxLineSearchIters:
            raise RuntimeError('No acceptable Cauchy point found after maximum allowed line search iterations')

    return alpha, s


def bound_constrained_trust_region_minimize(objective, x, bounds, settings, callback=None):
    trSize = settings.tr_size
    triedNewPrecond = False
    
    gradient = objective.gradient

    g = gradient(x)
    o = objective.value(x)
    R = project(x - g, bounds) - x
    prevOptimality = np.linalg.norm(R)
    print("\nInitial objective, optimality = ", o, prevOptimality)

    # this could potentially return an unstable solution
    if is_converged(objective, x, 0.0, 0.0, prevOptimality, prevOptimality, 0,
                    trSize, settings):
        if callback: callback(x, objective)
        return x

    # Set guess for first cauchy point step length
    gHg = g@objective.hessian_vec(x, g)
    if gHg > 0:
        alpha = (g@g) / gHg
    else:
        alpha = trSize / np.linalg.norm(g)
    
    cumulativeSpgIters=0
    
    for i in range(settings.max_trust_iters):
        # minimize 0.5*(2*r + J_sd)*d = r + 0.5*dJd
        
        if settings.use_incremental_objective:
            incremental_objective = lambda d: 0.5*((g + objective.gradient(x+d)) @ d)
        else:
            incremental_objective = lambda d: objective.value(x+d) - o
        
        hess_vec_func = lambda v: objective.hessian_vec(x, v)

        if settings.use_preconditioned_inner_product_for_spg:
            mult_by_approx_hessian = objective.multiply_by_approx_hessian
        else:
            mult_by_approx_hessian = lambda x: x

        alpha, cauchyPoint = find_generalized_cauchy_point(x, g, hess_vec_func,
                                                           bounds, alpha, trSize, settings)

        s, modelObjective, modelOptimality, stepType, spgIters = solve_spg_subproblem(
            x, cauchyPoint, g, bounds, hess_vec_func,
            objective.apply_precond, trSize, settings)
            
        cumulativeSpgIters += spgIters
        
        trSizeUsed = trSize
        
        y = x + s
        realObjective = incremental_objective(s)
        gy = gradient(y)
        R = project(y - gy, bounds) - y
        realOptimality = np.linalg.norm(R)

        # print('xk', y)
            
        if is_converged(objective, y, realObjective, modelObjective,
                        realOptimality, modelOptimality, spgIters, trSizeUsed,
                        settings):
            if callback: callback(y, objective)
            return y
        
        modelImprove = -modelObjective
        realImprove = -realObjective

        rho = realImprove / modelImprove

        if modelObjective > 0:
            print('Model objective increased.  Debug if you see this.')
            rho = realImprove / -modelImprove
            #exit(1)
            
        if not rho >= settings.eta2:  # write it this way to handle NaNs
            trSize *= settings.t1
        elif rho > settings.eta3 and is_on_boundary(stepType):
            trSize *= settings.t2

        willAccept = rho >= settings.eta1 or (rho >= 0 and realOptimality <= prevOptimality)
            
        print_min_banner(realObjective, modelObjective,
                         realOptimality, modelOptimality,
                         spgIters, trSizeUsed, stepType,
                         willAccept,
                         settings)

        if willAccept:
            x = y
            g = gy
            o = objective.value(x)
            prevOptimality = realOptimality
            triedNewPrecond = False
            if callback: callback(x, objective)

                
        if (spgIters >= settings.max_spg_iters
            or cumulativeSpgIters >= settings.max_cumulative_spg_iters):
            # TODO: Talk to Drew Kouri about preconditioning SPG
            # BT 11/2021
            # precond update suppressed because we don't use it
            # objective.update_precond(x)
            cumulativeSpgIters=0
 
        trSizeUsed = trSize
        
        if trSize < settings.min_tr_size:
            if not triedNewPrecond:
                print("The trust region is too small, updating precond and trying again.")
                objective.update_precond(x)
                cumulativeSpgIters=0
                triedNewPrecond = True
                trSize = settings.tr_size                    
            else:
                print("The trust region is still too small.  Accepting, but be careful.")
                if callback: callback(x, objective)
                return x
                    
    print("Reached the maximum number of trust region iterations.")
    if settings.check_stability:
        objective.check_stability(x)

        if callback: callback(x, objective)
    return x


def solve(objective, x0, p, lowerBounds, upperBounds, settings, callback=None,
          useWarmStart=True, updatePrecond=True):
    xBar0 = objective.scaling * x0
    lBar = objective.scaling * lowerBounds
    uBar = objective.scaling * upperBounds
    
    if useWarmStart:
        if updatePrecond:
            objective.update_precond(xBar0)
        
        dxBar = WarmStart.warm_start_increment(objective,
                                               xBar0, p)
        xBar0 += dxBar
        objective.p = p
    else:
        objective.p = p

    if updatePrecond:
        objective.update_precond(xBar0)

    bounds = np.column_stack((lBar, uBar))
        
    xBar = bound_constrained_trust_region_minimize(objective, xBar0, bounds, settings,
                                                   callback=callback)
    
    return objective.invScaling * xBar
    
