from optimism.JaxConfig import *
from optimism.treigen import treigen



from optimism.EquationSolver import Settings, get_settings, settings_with_new_tol, negCurveString, boundaryString, interiorString, is_converged, is_on_boundary, print_min_banner


class ModelProblem():

    def __init__(self, g):
        self.b = g
        self.v = []
        self.Kv = []
        self.isInitialized=False


    def is_initialized(self):
        return self.isInitialized

    
    def add_vector(self, v, Kv):
        N = len(self.v)
        for i in range(N):
            alpha = v@self.v[i]
            v = v - alpha*self.v[i]
            Kv = Kv - alpha*self.Kv[i]

        vnorm = np.linalg.norm(v)
        self.v.append(v/vnorm)
        self.Kv.append(Kv/vnorm)


    def setup_system(self):
        self.isInitialized=True
        # for the reduced system
        N = len(self.v)

        H = np.zeros((N,N))
        g = np.zeros((N))
        
        for i in range(N):
            g = g.at[i].set( self.v[i]@self.b )
            # g[i] = self.v[i]@self.b
            for j in range(N):
                H = H.at[i,j].set( self.v[i]@self.Kv[j] )
            # H[i,j] = vi^T K vj

        self.g = g
        self.H = 0.5*(H+H.T)

        
        
    def solve(self, Delta):
        # s below is alpha, the coefficeints for the vectors
        # min 1/2 alpha^T H alpha + alpha dot self.g, st. norm(alpha) <= Delta
        s = treigen.solve(self.H, self.g, Delta)
        print('chosen vector weights are = ', s)
        N = len(self.v)
        step = s[0]*self.v[0]
        for i in range(1,N):
            step = step + s[i]*self.v[i]
        return step
            
                

def project_to_boundary_with_coefs(z, d, trSize, zz, zd, dd):
    # find tau s.t. (z + tau*d)^2 = trSize^2
    tau = (np.sqrt( (trSize**2-zz)*dd + zd**2 ) - zd)/dd
    return z + tau*d


def trust_region_cg(x, r, Pr, HPr, hess_vec_func, precond, trSize, settings):
    # minimize r@z + 0.5*z@J@z
    z = 0.*x

    cgInexactRelTol = settings.cg_inexact_solve_ratio
    cgTolSquared = max(settings.cg_tol**2, cgInexactRelTol*cgInexactRelTol*r@r)
    if r@r < cgTolSquared:
        print('Entering cg with zero gradient, probably should not be getting here, please debug.')
        return z, interiorString, 0
    
    d = -Pr
    rPr = r@Pr
    curvature = -d@HPr
    
    usePrecondForStep = False
    
    for i in range(settings.max_cg_iters):
        alpha = rPr / curvature           
        zNp1 = z + alpha*d

        if curvature <= 0:
            zz = np.dot(z, z)
            zd = np.dot(z, d)
            dd = np.dot(d, d)
            zOut = project_to_boundary_with_coefs(z, d, trSize,
                                                  zz, zd, dd)
            return zOut, negCurveString, i+1
            
        if np.dot(zNp1, zNp1) > trSize**2:
            zz = np.dot(z, z)
            zd = np.dot(z, d)
            dd = np.dot(d, d)
            zOut = project_to_boundary_with_coefs(z, d, trSize,
                                                  zz, zd, dd)
            return zOut, boundaryString, i+1

        z = zNp1
           
        r += alpha * hess_vec_func(d)
        Pr = precond(r)
        rPrNp1 = r@Pr
          
        if r@r < cgTolSquared:
            return z, interiorString, i+1

        beta = rPrNp1 / rPr
        d = -Pr + beta*d
        rPr = rPrNp1
        curvature = d@( hess_vec_func(d) )
        
    return z, interiorString+'_', i+1



def spectral_gradient_minimize(x, g, hess_vec_func, trSize, settings):

    Hx = hess_vec_func(x)
    d = g + Hx
    energy = g@x + 0.5*x@Hx
    
    dNorm = np.linalg.norm(d)
    
    L = settings.over_iters

    lamMin = 1e-8
    lamMax = 1e8
    lam = 1.0

    for l in range(L):
        s = -lam * d
        h = hess_vec_func(s)
        alpha = 1.0
        hs = h@s
        ds = d@s
        if hs > 0:
            alpha = min(1.0, -ds/hs)
        else:
            print('negative curvature found')
            
        x += alpha*s
        d += alpha*h
        lam = max(lamMin, min(lamMax, s@s / hs))

    finalEnergy = g@x + 0.5*x@hess_vec_func(x)

    print('spg initial and final energy = ', energy, finalEnergy)
    
    return x


def trust_region_subspace_minimize(objective, x, settings, callback=None):
    
    trSize = settings.tr_size
    
    g = objective.gradient(x)
    o = objective.value(x)
    gNorm = np.linalg.norm(g)
    print("\nInitial objective, residual = ", o, gNorm)

    # this could potentially return an unstable solution
    if is_converged(objective, x, 0.0, 0.0, g, g, 0, trSize, settings):
        if callback: callback(x, objective)
        return x

    cumulativeCgIters=0
    
    for i in range(settings.max_trust_iters):
        # minimize 0.5*(2*r + J_sd)*d = r + 0.5*dHd

        trSizeForStep = trSize
        
        if settings.use_incremental_objective:
            incremental_objective = lambda d: 0.5*((g + objective.gradient(x+d)) @ d)
        else:
            incremental_objective = lambda d: objective.value(x+d) - o

        hess_vec_func = lambda v: objective.hessian_vec(x, v)



        # minimize z^T K z + g^z, z = alpha1*v1 + alpha2*v2 + ...
        subspace = ModelProblem(g)
                
        Pg = objective.apply_precond(g)
        HPg = hess_vec_func(Pg)
        
        d, stepType, cgIters = \
            trust_region_cg(x, g,
                            Pg, HPg,
                            hess_vec_func,
                            objective.apply_precond,
                            trSize, settings)
        cumulativeCgIters += cgIters

        trySpg = settings.over_iters > 0 and stepType !=interiorString
        if trySpg:
            d_spg = spectral_gradient_minimize(d, g, hess_vec_func,
                                               2*trSize, settings)

        Hd = hess_vec_func(d)
        dHd = d @ Hd
        modelObjective = g@d + 0.5*dHd
        
        if stepType != interiorString:
            subspace.add_vector(g, hess_vec_func(g))
            subspace.add_vector(Pg, HPg)
            if cgIters>1:
                subspace.add_vector(d, Hd)
            if trySpg:
                subspace.add_vector(d_spg, hess_vec_func(d_spg))
            subspace.setup_system()
                
            # try to find a better model solution
            d_ss = subspace.solve(trSizeForStep)
            Hd_ss = hess_vec_func(d_ss)
            modelObjective_ss = g@d_ss + 0.5*d_ss@Hd_ss

            #print('default, subspace objective = ', modelObjective, modelObjective_ss)
            
            if modelObjective_ss <= modelObjective:
                modelObjective = modelObjective_ss
                d = d_ss
            
        y = x+d
        realObjective = incremental_objective(d)
        gy = objective.gradient(y)

        if is_converged(objective, y, realObjective, modelObjective,
                        gy, g + Hd, cgIters, trSizeForStep, settings):
            if callback: callback(y, objective)
            return y
        
        modelImprove = -modelObjective
        realImprove = -realObjective
        rho = realImprove / modelImprove
        
        if modelObjective > 0:
            print('Found a positive model objective increase.  Debug if you see this.')
            rho = 0.0

        realResNorm = np.linalg.norm(gy)

        # accepts if the model is good or if the energy doesn't change, but the residual drops
        willAccept = rho >= settings.eta1 or (rho >= -0 and realResNorm <= gNorm)

        modelRes = g + Hd
        modelResNorm = np.linalg.norm(modelRes)
        print_min_banner(realObjective, modelObjective,
                         realResNorm, modelResNorm,
                         cgIters, trSizeForStep, stepType,
                         willAccept,
                         settings)
        
        if not rho >= settings.eta2: # this way to handle NaNs
            if willAccept:
                trSize *= settings.t1
            else: # recursively shrink and solve sub-space

                if not subspace.is_initialized():
                    subspace.add_vector(g, hess_vec_func(g))
                    subspace.add_vector(Pg, HPg)
                    if cgIters>1:
                        subspace.add_vector(d, Hd)
                    subspace.setup_system()

                stepType = boundaryString
                maxTrShrinks=10
                for i in range(maxTrShrinks):
                    trSizeForStep *= settings.t1
                    #print("Shrinking trust-region and resolving sub-space with size = ", trSizeForStep)
                    d = subspace.solve(trSizeForStep)
                    Hd = hess_vec_func(d)
                    modelObjective = g@d + 0.5*d@Hd

                    y = x+d
                    realObjective = incremental_objective(d)
                    gy = objective.gradient(y)
                    realResNorm = np.linalg.norm(gy)
                    
                    modelImprove = -modelObjective
                    realImprove = -realObjective
                    rho = realImprove / modelImprove
                    
                    willAccept = rho >= settings.eta1

                    modelRes = g + Hd
                    modelResNorm = np.linalg.norm(modelRes)
                    print_min_banner(realObjective, modelObjective,
                                     realResNorm, modelResNorm,
                                     0, trSizeForStep, stepType,
                                     willAccept,
                                     settings)
                    
                    if willAccept:
                        break
                        
                if willAccept:
                    trSize = trSizeForStep
                else:
                    # the subspace failed, force preconditioner update
                    print('subspace trust region reduction failed, trying the slow way and updating preconditioner')
                    trSize *= settings.t1
                    cumulativeCgIters = settings.max_cumulative_cg_iters
        
        elif rho > settings.eta3 and is_on_boundary(stepType):
            trSize *= settings.t2

        if willAccept:
            x = y
            g = gy
            o = objective.value(x)
            gNorm = realResNorm
            if callback: callback(x, objective)
            
        if cgIters >= settings.max_cg_iters or cumulativeCgIters >= settings.max_cumulative_cg_iters: 
            objective.update_precond(x)
            cumulativeCgIters=0
 
        
        if trSize < settings.min_tr_size:
            print("The trust region is too small.  Accepting, but be careful.")
            if callback: callback(x, objective)
            return x
                    
    print("Reached the maximum number of trust region iterations.")
    if settings.check_stability:
        objective.check_stability(x)

        if callback: callback(x, objective)
    return x

