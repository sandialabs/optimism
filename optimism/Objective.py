from optimism.FunctionSpace import DofManager
from optimism.JaxConfig import *
from optimism.SparseCholesky import SparseCholesky
from scipy.sparse import diags as sparse_diags
from scipy.sparse import csc_matrix
from typing import Optional
import equinox as eqx
import numpy as onp


# static vs dynamics
# differentiable vs undifferentiable
# TODO fix some of these type hints for better clarity. 
# maybe this will help formalize what's what when
class Params(eqx.Module):
    bc_data: any
    state_data: any
    design_data: any
    app_data: any
    time: any
    dynamic_data: any
    # Need the eqx.field(static=True) since DofManager
    # is composed of mainly og numpy arrays which leads
    # to the error
    #jax.errors.NonConcreteBooleanIndexError: Array boolean indices must be concrete; got ShapedArray(bool[x,x])
    dof_manager: DofManager = eqx.field(static=True)

    def __init__(
        self, 
        bc_data = None, 
        state_data = None, 
        design_data = None, 
        app_data = None, 
        time = None, 
        dynamic_data = None, 
        dof_manager: Optional[DofManager] = None
    ): 
        self.bc_data = bc_data
        self.state_data = state_data
        self.design_data = design_data
        self.app_data = app_data
        self.time = time
        self.dynamic_data = dynamic_data
        self.dof_manager = dof_manager

    def __getitem__(self, index):
        if index == 0:
            return self.bc_data
        elif index == 1:
            return self.state_data
        elif index == 2:
            return self.design_data
        elif index == 3:
            return self.app_data
        elif index == 4:
            return self.time
        elif index == 5:
            return self.dynamic_data
        elif index == 6:
            return self.dof_manager
        else:
            raise ValueError(f'Bad index value {index}')


# written for backwards compatability
# we can just use the eqx.tree_at syntax in simulations
# or we could write a single method bound to Params for this...
def param_index_update(p, index, newParam):
    if index == 0:
        p = eqx.tree_at(lambda x: x.bc_data, p, newParam)
    elif index == 1:
        p = eqx.tree_at(lambda x: x.state_data, p, newParam)
    elif index == 2:
        p = eqx.tree_at(lambda x: x.design_data, p, newParam)
    elif index == 3:
        p = eqx.tree_at(lambda x: x.app_data, p, newParam)
    elif index == 4:
        p = eqx.tree_at(lambda x: x.time, p, newParam)
    elif index == 5:
        p = eqx.tree_at(lambda x: x.dynamic_data, p, newParam)
    elif index == 6:
        p = eqx.tree_at(lambda x: x.dof_manager, p, newParam)
    else:
        raise ValueError(f'Bad index value {index}')

    return p


class PrecondStrategy:

    def __init__(self, objective_precond):
        self.objective_precond = objective_precond
        
        
    def initialize(self, x, p):
        self.K = self.objective_precond(x, p)
    
    
    def precond_at_attempt(self, attempt):
        if attempt==0:
            return self.K
        else:
            dAbs = onp.abs(self.K.diagonal())
            shift = pow(10, (-5+attempt))
            return self.K + sparse_diags( shift * dAbs, 0, format='csc' )


class TwoTryPrecondStrategy(PrecondStrategy):

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

        
    def initialize(self, x, p):
        self.x = x
        self.p = p

        
    def precond_at_attempt(self, attempt):
        if attempt==0:
            return self.f1(self.x, self.p)
        elif attempt==1:
            self.K = self.f2(self.x, self.p)
            return self.K
        else:
            dAbs = onp.abs(self.K.diagonal())
            shift = pow(10, (-5+attempt))
            return self.K + sparse_diags(shift * dAbs, format='csc')
        
    
class Objective:

    def __init__(self, f, x, p, precondStrategy=None):

        self.precond = SparseCholesky()
        self.precondStrategy = precondStrategy
        
        self.p = p
        
        self.objective=jit(f)
        self.grad_x = jit(grad(f,0))
        self.grad_p = jit(grad(f,1))
        
        self.hess_vec   = jit(lambda x, p, vx:
                              jvp(lambda z: self.grad_x(z,p), (x,), (vx,))[1])

        self.vec_hess   = jit(lambda x, p, vx:
                              vjp(lambda z: self.grad_x(z,p), x)[1](vx))
        
        self.jac_xp_vec = jit(lambda x, p, vp0:
                              jvp(lambda q0: self.grad_x(x, param_index_update(p,0,q0)),
                                  (p[0],),
                                  (vp0,))[1])

        self.jac_xp2_vec = jit(lambda x, p, vp2:
                               jvp(lambda q2: self.grad_x(x, param_index_update(p,2,q2)),
                                   (p[2],),
                                   (vp2,))[1])
        
        self.vec_jac_xp0 = jit(lambda x, p, vx:
                               vjp(lambda q0: self.grad_x(x, param_index_update(p,0,q0)), p[0])[1](vx))
        
        self.vec_jac_xp1 = jit(lambda x, p, vx:
                               vjp(lambda q1: self.grad_x(x, param_index_update(p,1,q1)), p[1])[1](vx))
        
        self.vec_jac_xp2 = jit(lambda x, p, vx:
                               vjp(lambda q2: self.grad_x(x, param_index_update(p,2,q2)), p[2])[1](vx))

        self.vec_jac_xp4 = jit(lambda x, p, vx:
                               vjp(lambda q4: self.grad_x(x, param_index_update(p,4,q4)), p[4])[1](vx))


        self.grad_and_tangent = lambda x, p: linearize(lambda z: self.grad_x(z,p), x)
        
        self.hess = jit(jacfwd(self.grad_x, 0))

        self.scaling = 1.0
        self.invScaling = 1.0
        
        
    def value(self, x):
        return self.objective(x, self.p)

    def gradient(self, x):
        return self.grad_x(x, self.p)
    
    def gradient_p(self, x):
        return self.grad_p(x, self.p)

    def hessian_vec(self, x, vx):
        return self.hess_vec(x, self.p, vx)

    def gradient_and_tangent(self, x):
        return self.grad_and_tangent(x, self.p)
    
    def vec_hessian(self, x, vx):
        return self.vec_hess(x, self.p, vx)
                              
    def hessian(self, x):
        return self.hess(x, self.p)

    def jacobian_p_vec(self, x, vp):
        return self.jac_xp_vec(x, self.p, vp)

    def jacobian_p2_vec(self, x, vp):
        return self.jac_xp2_vec(x, self.p, vp)

    def vec_jacobian_p0(self, x, vp):
        return self.vec_jac_xp0(x, self.p, vp)
    
    def vec_jacobian_p1(self, x, vp):
        return self.vec_jac_xp1(x, self.p, vp)

    def vec_jacobian_p2(self, x, vp):
        return self.vec_jac_xp2(x, self.p, vp)

    def vec_jacobian_p4(self, x, vp):
        return self.vec_jac_xp4(x, self.p, vp)
        
    def apply_precond(self, vx):
        if self.precond:
            return self.precond.apply(vx)
        else:
            return vx

    def multiply_by_approx_hessian(self, vx):
        if self.precond:
            return self.precond.multiply_by_approximate(vx)
        else:
            return vx

    def update_precond(self, x):
        if self.precondStrategy==None:
            print('Updating with dense preconditioner in Objective.')
            K = csc_matrix(self.hessian(x))
            def stiffness_at_attempt(attempt):
                if attempt==0:
                    return K
                else:
                    dAbs = onp.abs(K.diagonal())
                    shift = pow(10, (-5+attempt))
                    return K + sparse_diags(shift * dAbs, 0, format='csc')
            self.precond.update(stiffness_at_attempt)
        else:
            self.precondStrategy.initialize(x, self.p)
            self.precond.update(self.precondStrategy.precond_at_attempt)

    def check_stability(self, x):
        if self.precond:
            self.precond.check_stability(x, self.p)


class ScaledPrecondStrategy(PrecondStrategy):

    def __init__(self,
                 precondStrategy,
                 dofScaling):
        self.ps = precondStrategy
        self.invScaling = sparse_diags(onp.array(dofScaling), format='csc')

    
    def initialize(self, x, p):        
        self.ps.initialize(self.invScaling*x, p)

        
    def precond_at_attempt(self, attempt):
        K = self.ps.precond_at_attempt(attempt)
        K2 = csc_matrix( self.invScaling.T * K * self.invScaling )
        
        Kdiag = np.array(K2.diagonal())

        print('min, max diagonal stiffness = ',
              np.min(Kdiag),
              np.max(Kdiag))
        return K2


class ScaledObjective(Objective):
    
    def __init__(self,
                 objective_func,
                 x0,
                 p,
                 precondStrategy=None):
        
        if precondStrategy:
            precondStrategy.initialize(x0, p)
            K0 = precondStrategy.precond_at_attempt(0)
            scaling = np.sqrt(K0.diagonal())
            invScaling = 1.0/scaling
            
            scaledPrecondStrategy = ScaledPrecondStrategy(precondStrategy,
                                                          invScaling)

        else:
            scaling = 1.0
            invScaling = 1.0
            scaledPrecondStrategy = None
            
        def scaled_objective(xBar, p):
            x = invScaling * xBar
            return objective_func(x, p)

        xBar0 = scaling * x0
        super().__init__(scaled_objective,
                         xBar0,
                         p,
                         scaledPrecondStrategy)

        self.scaling = scaling
        self.invScaling = invScaling
        

    def get_value(self, x):
        return self.value(self.scaling * x)
    

    def get_residual(self, x):
        return self.gradient(self.scaling * x)

    
