from optimism.JaxConfig import *
import optimism.EquationSolver as EquationSolver
from optimism.Objective import Objective
from optimism.Objective import param_index_update
from optimism.SparseCholesky import SparseCholesky
import numpy as onp
from scipy.sparse import csc_matrix
from scipy.sparse import diags as sparse_diags


def fischer_burmeister(c, l, k):
    ck = c*k
    return np.sqrt(ck**2 + l**2) - ck - l


def fischer_burmeister_jac_l(c, l, k):
    t = np.sqrt((c*k)**2 + l**2)
    return (l - t) / t


class PrecondStrategy:

    def __init__(self, objective_precond, constraint_precond):
        self.objective_precond = objective_precond
        self.constraint_precond = constraint_precond
        
        
    def initialize(self, x, p, lam, kappa):
        self.K = self.objective_precond(x, p)
        if self.constraint_precond:
            self.K += self.constraint_precond(x, p, lam, kappa)
    
    
    def precond_at_attempt(self, attempt):
        if attempt==0:
            return self.K
        else:
            dAbs = onp.abs(self.K.diagonal())
            shift = pow(10, (-5+attempt))
            return self.K + sparse_diags( shift * dAbs, 0, format='csc' )

            
class ConstrainedObjective(Objective):
    
    def __init__(self,
                 objective_func, constraint_func,
                 x, p, lam, kappa, precondStrategy=None):

        self.precond = SparseCholesky()
        self.precondStrategy = precondStrategy

        # cannot update this on the fly, it is baked into the jits below
        self.constraintKappa = np.array(kappa)

        self.p = p
        self.lam = lam
        self.kappa = kappa

        f = self.create_augmented_lagrangian(objective_func, constraint_func)
                
        self.jit_objective=jit(f)
        grad_x = grad(f,0)

        self.jit_grad_x = jit(grad_x)
        self.jit_grad_p = jit(grad(f,1))
        self.jit_grad_l = jit(grad(f,2))

        self.jit_hess = jit(hessian(f,0))
        
        self.jit_hess_vec = jit(lambda x, p, l, k, vx:
                                jvp(lambda z: grad_x(z,p,l,k), (x,), (vx,))[1])

        # only jac-vec wrt first parameter in parameter tuple
        self.jit_jac_xp_vec = jit(lambda x, p, l, k, vp:
                                  jvp(lambda q0: grad_x(x,param_index_update(p,0,q0),l,k),
                                      (p[0],),
                                      (vp,))[1])
        
        self.jit_jac_xl_vec = jit(lambda x, p, l, k, vl:
                                  jvp(lambda j: grad_x(x,p,j,k), (l,), (vl,))[1])
        
        self.jit_constraint_func = jit(constraint_func)
        
        def ncp_func(x, p, l):
            c = constraint_func(x, p)
            return vmap(fischer_burmeister)(c, l, self.constraintKappa)

        self.jit_ncp_func = jit(ncp_func)

        def ncp_hess(x, p, l):
            c = constraint_func(x, p)
            return vmap(fischer_burmeister_jac_l)(c, l, self.constraintKappa)

        self.jit_ncp_hess = jit(ncp_hess)

        def hres_all_args(x, p, l, k):
            return np.hstack( (grad_x(x,p,l,k),
                               ncp_func(x,p,l) ) )

        self.hres = lambda xl, p, k: hres_all_args(xl[:-k.size], p, xl[-k.size:], k)
        self.jit_hres = jit(self.hres)
                
        self.jit_hres_jac_vec = jit(lambda xl, p, k, vxl:
                                    jvp(lambda zl: self.hres(zl, p, k), (xl,), (vxl,))[1])
        
        # only jac-vec wrt first parameter in parameter tuple
        self.jit_hres_jac_p_vec = jit(lambda xl, p, k, vp:
                                      jvp(lambda q0: self.hres(xl,param_index_update(p,0,q0),k),
                                          (p[0],),
                                          (vp,))[1])


    def create_augmented_lagrangian(self, objective_func, constraint_func):
        def f(x, p, l, k):
            c = constraint_func(x, p)
            penalty = np.where( l >= k*c,
                                -c*l + 0.5*k*c*c,
                                -0.5*l*l/k )
            return objective_func(x, p) + np.sum(penalty)
        return f
        
    
    def value(self, x):
        return self.jit_objective(x, self.p, self.lam, self.kappa)

    
    def gradient(self, x):
        return self.jit_grad_x(x, self.p, self.lam, self.kappa)

    
    def gradient_p(self, x):
        return self.jit_grad_p(x, self.p, self.lam, self.kappa)

    
    def gradient_l(self, x):
        return self.jit_grad_l(x, self.p, self.lam, self.kappa)

    
    def hessian(self, x):
        return self.jit_hess(x, self.p, self.lam, self.kappa)

    
    def hessian_vec(self, x, vx):        
        return self.jit_hess_vec(x, self.p, self.lam, self.kappa, vx)

    
    def jacobian_p_vec(self, x, vp):
        return self.jit_jac_xp_vec(x, self.p, self.lam, self.kappa, vp)

    
    def jacobian_l_vec(self, x, vl):
        return self.jit_jac_xl_vec(x, self.p, self.lam, self.kappa, vl)

    
    def constraint(self, x):
        return self.jit_constraint_func(x, self.p)

    
    def ncp(self, x):
        return self.jit_ncp_func(x, self.p, self.lam)

    
    def ncp_hessian(self, x):
        return self.jit_ncp_hess(x, self.p, self.lam)

    # higher order lagrange multiplier update related functions
            
    def constrained_residual(self, xl):
        return self.jit_hres(xl, self.p, self.kappa)

    
    def constrained_jacobian_vec(self, xl, vxl):
        return self.jit_hres_jac_vec(xl, self.p, self.kappa, vxl)

    
    def constrained_jacobian_p_vec(self, xl, vp):
        return self.jit_hres_jac_p_vec(xl, self.p, self.kappa, vp)

    
    def total_residual(self, x):
        return self.constrained_residual(np.hstack((x,self.lam)))


    def update_precond(self, x):
        if self.precondStrategy==None:
            print('Updating with dense preconditioner in ConstrainedObjective.')
            K = csc_matrix(self.hessian(x))
            d = np.array(K.diagonal())
            
            def stiffness_at_attempt(attempt):
                if attempt==0:
                    return K
                else:
                    dAbs = onp.abs(K.diagonal())
                    shift = pow(10, (-5+attempt))
                    return K + sparse_diags(shift * dAbs, 0, format='csc')
                
            self.precond.update(stiffness_at_attempt)
        else:
            self.precondStrategy.initialize(x, self.p, self.lam, self.kappa)
            self.precond.update(self.precondStrategy.precond_at_attempt)


    def reset_kappa(self):
        self.kappa = np.array(self.constraintKappa)
            

class ConstrainedQuasiObjective(ConstrainedObjective):

    def create_augmented_lagrangian(self, objective_func, constraint_func):
        def f(x, p, l, k):
            c = constraint_func(x, p)
            penalty = np.where( l > k*c,
                                -c*l + 0.5*k*c*c,
                                -0.5*l*l/k )
            return objective_func(x, l, p) + np.sum(penalty)
        return f
    
    
    def __init__(self,
                 objective_func, constraint_func,
                 x, p, lam, kappa, precondStrategy=None):

        super().__init__(objective_func, constraint_func,
                         x, p, lam, kappa, precondStrategy)
    
