from optimism.JaxConfig import *
import optimism.LU as LU
from optimism.Timer import Timer
from optimism import WarmStart
from scipy.sparse import save_npz
from scipy.sparse.linalg import LinearOperator, gmres


Settings = namedtuple('Settings', ['t1','t2','eta1','eta2','eta3','max_trust_iters','tol','max_cg_iters','max_cumulative_cg_iters','cg_tol','cg_inexact_solve_ratio','tr_size','min_tr_size','check_stability','use_preconditioned_inner_product_for_cg','use_incremental_objective','debug_info','over_iters'])

#must have cg_tol < tol
# consider changing to 0.5, 1.75, 1e-9, 0.1, 0.75
def get_settings(t1=0.25, t2=1.75, eta1=1e-10, eta2=0.1, eta3=0.5,
                 max_trust_iters=100,
                 tol=1e-8,
                 max_cg_iters=50,
                 max_cumulative_cg_iters=1000,
                 cg_tol=None,
                 cg_inexact_solve_ratio=1e-5,
                 tr_size=2.0,
                 min_tr_size=1e-8,
                 check_stability=False,
                 use_preconditioned_inner_product_for_cg=False,
                 use_incremental_objective=False,
                 debug_info=True,
                 over_iters=0):
    if cg_tol==None:
        cg_tol = 0.2*tol

    return Settings(t1, t2, eta1, eta2, eta3,
                    max_trust_iters=max_trust_iters,
                    tol=tol,
                    max_cg_iters=max_cg_iters,
                    max_cumulative_cg_iters=max_cumulative_cg_iters,
                    cg_tol=cg_tol,
                    cg_inexact_solve_ratio=cg_inexact_solve_ratio,
                    tr_size=tr_size,
                    min_tr_size=min_tr_size,
                    check_stability=check_stability,
                    use_preconditioned_inner_product_for_cg=use_preconditioned_inner_product_for_cg,
                    use_incremental_objective=use_incremental_objective,
                    debug_info=debug_info,
                    over_iters=over_iters)


def settings_with_new_tol(oldSettings, newTol):
    newSettings = Settings(t1=oldSettings.t1,
                           t2=oldSettings.t2,
                           eta1=oldSettings.eta1,
                           eta2=oldSettings.eta2,
                           eta3=oldSettings.eta3,
                           max_trust_iters=oldSettings.max_trust_iters,
                           tol=newTol, # here
                           max_cg_iters=oldSettings.max_cg_iters,
                           max_cumulative_cg_iters=oldSettings.max_cumulative_cg_iters,
                           cg_tol=0.2*newTol, # here
                           cg_inexact_solve_ratio=oldSettings.cg_inexact_solve_ratio,
                           tr_size=oldSettings.tr_size,
                           min_tr_size=oldSettings.min_tr_size,
                           check_stability=oldSettings.check_stability,
                           use_preconditioned_inner_product_for_cg=oldSettings.use_preconditioned_inner_product_for_cg,
                           use_incremental_objective=oldSettings.use_incremental_objective,
                           debug_info=oldSettings.debug_info,
                           over_iters=oldSettings.over_iters)
                           

    return newSettings


negCurveString='neg curve'
boundaryString='boundary'
interiorString='interior'


def is_on_boundary(stepType):
    return stepType==boundaryString or stepType==negCurveString


def project_to_boundary(z, d, trSize, zz):
    # find tau s.t. (z + tau*d)^2 = trSize^2
    dd = np.dot(d,d)
    zd = np.dot(z,d)
    tau = (np.sqrt( (trSize**2-zz)*dd + zd**2 ) - zd)/dd        
    # is it ever better to choose the - sqrt() branch?
    return z + tau*d


def preconditioned_project_to_boundary(z, d, trSize, zz, mult_by_approx_hessian):
    # find tau s.t. (z + tau*d)^2 = trSize^2
    Pd = mult_by_approx_hessian(d)
    dd = np.dot(d,Pd)
    zd = np.dot(z,Pd)
    tau = (np.sqrt( (trSize**2-zz)*dd + zd**2 ) - zd)/dd
    # is it ever better to choose the - sqrt() branch?
    return z + tau*d


def project_to_boundary_with_coefs(z, d, trSize, zz, zd, dd):
    # find tau s.t. (z + tau*d)^2 = trSize^2
    tau = (np.sqrt( (trSize**2-zz)*dd + zd**2 ) - zd)/dd
    return z + tau*d


def update_step_length_squared(alpha, zz, zd, dd):
    return zz + 2*alpha*zd + alpha*alpha*dd


def cg_inner_products_preconditioned(alpha, beta, zd, dd, rPr, z, d):
    # recurrence formulas from Gould et al. doi:10.1137/S1052623497322735
    zd = beta * ( zd + alpha*dd )
    dd = rPr + beta*beta*dd
    return zd, dd


def cg_inner_products_unpreconditioned(alpha, beta, zd, dd, rPr, z, d):
    zd = z @ d
    dd = d @ d
    return zd, dd


# returns increment of solution and True if the entire system is solved
def solve_trust_region_equality_constraint(x, g, J, trSize, settings):
    # minimuze 0.5*(J*z+g)^2
    
    r = J.multiply_by_transpose( J.solve_transpose(J.solve(g)) )
    d = -r
    z = 0.*x
    zz = 0.
    rr = np.dot(r,r)

    cgTolSquared = max(settings.cg_tol**2, 1e-10*rr)
    
    for i in range(settings.max_cg_iters):
        curvature = np.sum( (J.solve(J@d))**2 )

        if curvature <= 0:
            return z, negCurveString, i

        alpha = rr / curvature

        zNp1 = z + alpha*d
        zzNp1 = np.dot(zNp1,zNp1)
        if zzNp1 > trSize**2:
            return project_to_boundary(z, d, trSize, zz), boundaryString, i+1

        z = zNp1
        zz = zzNp1

        r += alpha * J.multiply_by_transpose(J.solve_transpose(J.solve(J@d)))
        rrNp1 = np.dot(r,r)
          
        if rrNp1 < cgTolSquared:
            return z, interiorString, i+1
        
        beta = rrNp1 / rr
        rr = rrNp1
        d = -r + beta*d
            
    return z, interiorString+'_', i+1


def solve_trust_region_minimization(x, r, hess_vec_func, precond, trSize, settings):
    # minimize r@z + 0.5*z@J@z
    z = 0.*x
    zz = 0.

    cgInexactRelTol = settings.cg_inexact_solve_ratio
    cgTolSquared = max(settings.cg_tol**2, cgInexactRelTol*cgInexactRelTol*r@r)
    if r@r < cgTolSquared:
        return z, z, interiorString, 0
    
    Pr = precond(r)
    d = -Pr
    cauchyP = np.array(d)
    
    rPr = r@Pr

    zz = 0.0
    zd = 0.0
    if settings.use_preconditioned_inner_product_for_cg:
        dd = rPr
        cg_inner_products = cg_inner_products_preconditioned
    else:
        dd = d @ d
        cg_inner_products = cg_inner_products_unpreconditioned
    
    for i in range(settings.max_cg_iters):
        curvature = d@( hess_vec_func(d) )
        alpha = rPr / curvature
           
        zNp1 = z + alpha*d
        zzNp1 = update_step_length_squared(alpha, zz, zd, dd)

        if curvature <= 0:
            zOut = project_to_boundary_with_coefs(z, d, trSize,
                                                  zz, zd, dd)
            return zOut, cauchyP, negCurveString, i+1

        if zzNp1 > trSize**2:
            zOut = project_to_boundary_with_coefs(z, d, trSize,
                                                  zz, zd, dd)
            return zOut, cauchyP, boundaryString, i+1

        z = zNp1 # z + alpha*d
           
        r += alpha * hess_vec_func(d)
        Pr = precond(r)
        rPrNp1 = r@Pr
          
        if r@r < cgTolSquared:
            return z, cauchyP, interiorString, i+1

        beta = rPrNp1 / rPr
        rPr = rPrNp1
        d = -Pr + beta*d

        zz = zzNp1
        zd, dd = cg_inner_products(alpha, beta, zd, dd, rPr, z, d)
            
    return z, cauchyP, interiorString+'_', i+1


# essentially deprecated
def print_banner(objective, modelObjective, cgIters, trSize, onBoundary, willAccept):
    willAccept = " True" if willAccept else "False"

    print('obj=', f"{objective:16.11}",
          ', model obj=', f"{modelObjective:16.11}",
          ', cg iters=', f"{cgIters:3}",
          ', tr size=', f"{trSize:12.6}",
          ', ', f"{onBoundary:9}",
          ', acceped=', willAccept)

    
def print_min_banner(objective, modelObjective, res, modelRes, cgIters, trSize, onBoundary, willAccept, settings):
    if settings.debug_info==False: return
    
    willAccept = " True" if willAccept else "False"
    
    print('obj=', f"{objective:15.9}",
          ', model obj=', f"{modelObjective:15.9}",
          ', res=', f"{res:15.9}", ', model res=', f"{modelRes:15.9}",
          ', cg iters=', f"{cgIters:3}",
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

    
def trust_region_least_squares_solve(objective, x, settings):
    trSize = settings.tr_size

    equation = objective.gradient
    g = equation(x)
    o = 0.5*np.dot(g, g)
        
    J = LU.LU(objective.hessian(x))
    
    for i in range(settings.max_trust_iters):

        dx, stepType, cgIters = solve_trust_region_equality_constraint(x, g, J, trSize, settings)
        
        modelObjective = 0.5*np.sum( (J@dx + g)**2 )
        
        y = x+dx
        gy = equation(y)
        oy = 0.5*np.dot(gy, gy)

        if 2*oy < settings.tol**2:
            print_banner(oy, modelObjective, cgIters, trSize, stepType, True)
            if settings.check_stability:
                objective.check_stability(y)
            return y
        
        modelImprove = o - modelObjective
        realImprove = o - oy

        rho = realImprove / modelImprove

        if not (rho >= settings.eta2):
            trSize *= settings.t1
        elif rho > settings.eta3 and is_on_boundary(stepType):
            trSize *= settings.t2

        print_banner(oy, modelObjective, cgIters, trSize, stepType, rho >= settings.eta1)

        if trSize < settings.min_tr_size: break
        
        if rho >= settings.eta1:
            x = y
            g = gy
            o = oy
            J.update( objective.hessian(x) )


    print("Reached the maximum number of trust region iterations or trust region is too small.")
    if settings.check_stability:
        objective.check_stability(x)
    return x


def is_converged(objective, x, realO, modelO, realRes, modelRes, cgIters, trSize, settings):
    gg = realRes@realRes
    if gg < settings.tol**2:
        modelResNorm = np.linalg.norm(modelRes)
        realResNorm = np.sqrt(gg)
        print_min_banner(realO, modelO,
                         realResNorm,
                         modelResNorm,
                         cgIters,
                         trSize,
                         interiorString,
                         True,
                         settings)
        if settings.check_stability:
            objective.check_stability(x)

        print('') # a bit of output formatting
            
        return True
    return False
        

def dogleg_step(cp, newtonP, trSize, mat_mul):
    cc = cp@mat_mul(cp)
    nn = newtonP@mat_mul(newtonP)
    tt = trSize*trSize
    
    if cc >= tt: #return cauchy point if it extends outside the tr
        #print('cp on boundary')
        return cp * np.sqrt(tt/cc)

    if cc > nn: # return cauchy point?  seems the preconditioner was not accurate?
        print('cp outside newton, preconditioner likely inaccurate')
        return cp

    if nn > tt: # on the dogleg (we have nn >= cc, and tt >= cc)
        #print('dogleg')
        return preconditioned_project_to_boundary(cp,
                                                  newtonP-cp,
                                                  trSize,
                                                  cc,
                                                  mat_mul)
    #print('quasi-newton step')
    return newtonP


def trust_region_minimize(objective, x, settings, callback=None):
    trSize = settings.tr_size
    triedNewPrecond = False
    
    gradientAndTanOpt = objective.gradient_and_tangent
    gradient = objective.gradient

    #g,hess_vec_func = gradientAndTanOpt(x)
    g = gradient(x)
    o = objective.value(x)
    gNorm = np.linalg.norm(g)
    print("\nInitial objective, residual = ", o, gNorm)

    # this could potentially return an unstable solution
    if is_converged(objective, x, 0.0, 0.0, g, g, 0, trSize, settings):
        if callback: callback(x, objective)
        return x

    cumulativeCgIters=0
    
    for i in range(settings.max_trust_iters):
        # minimize 0.5*(2*r + J_sd)*d = r + 0.5*dJd
        
        if settings.use_incremental_objective:
            incremental_objective = lambda d: 0.5*((g + objective.gradient(x+d)) @ d)
        else:
            incremental_objective = lambda d: objective.value(x+d) - o
        
        hess_vec_func = lambda v: objective.hessian_vec(x, v)
        mult_by_approx_hessian = objective.multiply_by_approx_hessian if settings.use_preconditioned_inner_product_for_cg else lambda x: x

        gKg = g@hess_vec_func(g)
        if gKg > 0:
            alpha = -(g@g) / gKg
            cauchyPoint = alpha * g
            cauchyPointNormSquared = cauchyPoint@mult_by_approx_hessian(cauchyPoint)
        else:
            cauchyPoint =  -g * (trSize / np.sqrt(g@mult_by_approx_hessian(g)))
            cauchyPointNormSquared = trSize*trSize
            print('negative curvature unpreconditioned cauchy point direction found.')
            
        if cauchyPointNormSquared >= trSize*trSize:
            print('unpreconditioned gradient cauchy point outside trust region at dist = ', np.sqrt(cauchyPointNormSquared))
            cauchyPoint *= (trSize / np.sqrt(cauchyPointNormSquared))
            cauchyPointNormSquared = trSize*trSize
            qNewtonPoint = cauchyPoint
            stepType = boundaryString
            cgIters = 1
        else:
            qNewtonPoint, _, stepType, cgIters = \
                solve_trust_region_minimization(x, g,
                                                hess_vec_func,
                                                objective.apply_precond,
                                                trSize, settings)

            
        cumulativeCgIters += cgIters
        
        trSizeUsed = trSize
        happyAboutTrSize=False
        while not happyAboutTrSize:
            d = dogleg_step(cauchyPoint, qNewtonPoint, trSize, mult_by_approx_hessian)

            Jd = hess_vec_func(d)
            dJd = d @ Jd
            modelObjective = g @ d + 0.5*dJd
            
            y = x+d
            realObjective = incremental_objective(d)
            gy = gradient(y)

            if is_converged(objective, y, realObjective, modelObjective,
                            gy, g + Jd, cgIters, trSizeUsed, settings):
                if callback: callback(y, objective)
                return y
        
            modelImprove = -modelObjective
            realImprove = -realObjective

            rho = realImprove / modelImprove

            if modelObjective > 0:
                print('Found a positive model objective increase.  Debug if you see this.')
                rho = realImprove / -modelImprove
                #exit(1)
            
            if not rho >= settings.eta2:  # write it this way to handle NaNs
                trSize *= settings.t1
            elif rho > settings.eta3 and is_on_boundary(stepType):
                trSize *= settings.t2

            modelRes = g + Jd
            modelResNorm = np.linalg.norm(modelRes)
            realResNorm = np.linalg.norm(gy)

            willAccept = rho >= settings.eta1 or (rho >= -0 and realResNorm <= gNorm)
            
            print_min_banner(realObjective, modelObjective,
                             realResNorm, modelResNorm,
                             cgIters, trSizeUsed, stepType,
                             willAccept,
                             settings)

            if willAccept:
                x = y
                g = gy
                #g,hess_vec_func = gradientAndTanOpt(x) 
                o = objective.value(x)
                gNorm = realResNorm
                triedNewPrecond = False
                happyAboutTrSize = True

                if callback: callback(x, objective)
            else:
                # set these for output
                # trust region will continue to strink until we find a solution on the boundary
                stepType=boundaryString
                cgIters = 0
                
            if cgIters >= settings.max_cg_iters or cumulativeCgIters >= settings.max_cumulative_cg_iters: 
                objective.update_precond(x)
                cumulativeCgIters=0
 
            trSizeUsed = trSize
        
            if trSize < settings.min_tr_size:

                if not triedNewPrecond:
                    print("The trust region is too small, updating precond and trying again.")
                    objective.update_precond(x)
                    cumulativeCgIters=0
                    triedNewPrecond = True
                    happyAboutTrSize = True
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


def newton(objective, x, settings, callback=None):

    res_func = objective.gradient
    hess_func = objective.hessian
    
    sz = x.size
    precondOp = LinearOperator((sz,sz),
                               lambda v: objective.apply_precond(v))
    
    for i in range(settings.max_trust_iters):
        print('Newton step = ', i)

        if callback: callback(x, objective.p)
        
        sparse=True
        if sparse:
            def apply_a(v):
                return objective.hessian_vec(x, v)
            
            aOp = LinearOperator((sz,sz),
                                 lambda v: apply_a(v))
            
            g = res_func(x)
            gNorm = np.linalg.norm(g)

            print('norm = ', gNorm)
            
            dx, exitcode = gmres(aOp, g, tol=1e-3*gNorm, atol=0, M=precondOp, maxiter=settings.max_cg_iters)

            gTrial = np.linalg.norm( res_func(x-dx) )
            print('trial norm  = ', gTrial)
            while gTrial > gNorm:
                dx *= 0.25
                gTrial = np.linalg.norm( res_func(x-dx) )
                print('trial norm  = ', gTrial)
                
            x -= dx
        
        else:
            g = res_func(x)
            H = hess_func(x)
            x -= np.linalg.solve(H,g)
            
        o = objective.value(x)
        if is_converged(objective, x, o, o,
                        g, g, 1, np.inf, settings):
            return x

    print("Reached the maximum number of newton iterations.")
    if settings.check_stability:
        objective.check_stability(x)
    return x


# This solve uses a different interface.  There is no globalization yet

def newton_solve(objective_func, solution, maxSteps=1):
    grad_func = grad(objective_func)
    hess_func = jacfwd(grad_func)

    for step in range(maxSteps):
        with Timer(name="Step"):
            with Timer(name="Fill Grad"):
                gradients = grad_func(solution)
                print('res = ', np.linalg.norm(gradients))
        
            with Timer(name="Fill Hess"):
                hess = hess_func(solution)
    
            nDofs = gradients.size
            gradShape = gradients.shape
    
            gradients = np.reshape(gradients, (nDofs,))
            
            hess = np.reshape(hess, (nDofs, nDofs))
            
            with Timer(name="LinearSolve"):
                solution = solution - np.reshape(np.linalg.solve(hess,gradients), gradShape)
                
    return solution


def nonlinear_equation_solve(objective, x0, p, settings,
                             solver_algorithm=trust_region_minimize,
                             callback=None,
                             useWarmStart=True,
                             updatePrecond=True):
    xBar0 = objective.scaling * x0
    
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
        
    xBar = solver_algorithm(objective, xBar0, settings, callback=callback)
    
    return objective.invScaling * xBar
    
