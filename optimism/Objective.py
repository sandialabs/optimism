from optimism.JaxConfig import *
from optimism.SparseCholesky import SparseCholesky
import numpy as onp
import jax.numpy as np
import jax
from jax import jit, grad, jacfwd, jvp, vjp
from scipy.sparse import csc_matrix
from scipy.sparse import diags as sparse_diags 


# static vs dynamics
# differentiable vs undifferentiable
Params = namedtuple('Params',
                    ['bc_data',
                     'state_data',
                     'design_data',
                     'app_data',
                     'time',
                     'dynamic_data',
                     'prop_data'],
                     defaults=(None,None,None,None,None,None,None))


def param_index_update(p, index, newParam):
    if index==0:
        return Params(newParam, p[1], p[2], p[3], p[4], p[5], p[6])
    if index==1:
        return Params(p[0], newParam, p[2], p[3], p[4], p[5], p[6])
    if index==2:
        return Params(p[0], p[1], newParam, p[3], p[4], p[5], p[6])
    if index==3:
        return Params(p[0], p[1], p[2], newParam, p[4], p[5], p[6])
    if index==4:
        return Params(p[0], p[1], p[2], p[3], newParam, p[5], p[6])
    if index==5:
        return Params(p[0], p[1], p[2], p[3], p[4], newParam, p[6])
    if index==6:
        return Params(p[0], p[1], p[2], p[3], p[4], p[5], newParam)
    print('invalid index passed to param_index_update = ', index)


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

# Objective class for handling Multi-Point Constraints
class ObjectiveMPC(Objective):
    def __init__(self, objective_func, x_full, p, dofManagerMPC, precondStrategy=None):
        """
        ObjectiveMPC: wraps the original energy functional to solve only for reduced DOFs.
        - Applies u = T * ũ + s̃ for MPC condensation.
        - Automatically condenses gradient and Hessian.
        """
        self.dofManagerMPC = dofManagerMPC
        self.T = dofManagerMPC.T
        self.s_tilde = dofManagerMPC.s_tilde
        self.p = p
        self.precondStrategy = precondStrategy

        # Store full functional and derivatives
        self.full_objective_func = jit(objective_func)
        self.full_grad_x = jit(grad(objective_func, 0))
        self.full_hess = jit(jacfwd(self.full_grad_x, 0))

        self.scaling = 1.0
        self.invScaling = 1.0

        # Reduced initial guess from full vector
        self.x_reduced0 = self.reduce_to_independent_dofs(x_full)

        # Define reduced-space objective and gradient
        self.objective = jit(lambda x_red, p: self.full_objective_func(self.expand_to_full_dofs(x_red), p))
        self.grad_x = jit(lambda x_red, p: self.reduce_gradient(self.full_grad_x(self.expand_to_full_dofs(x_red), p)))

        self.precond = SparseCholesky()

    def reduce_to_independent_dofs(self, x_full):
        """Extract reduced DOFs from full vector."""
        return self.dofManagerMPC.get_unknown_values(x_full)

    def expand_to_full_dofs(self, x_reduced):
        """Reconstruct full DOF vector using u = T ũ + s̃."""
        return self.dofManagerMPC.create_field(x_reduced)

    def reduce_gradient(self, grad_full):
        """Project full gradient to reduced space."""
        return self.dofManagerMPC.get_unknown_values(grad_full)

    def value(self, x_reduced):
        """Return energy functional in reduced space."""
        return self.objective(x_reduced, self.p)

    def gradient(self, x_reduced):
        """Return reduced gradient."""
        return self.grad_x(x_reduced, self.p)

    def hessian(self, x_reduced):
        """Compute reduced Hessian H̃ = Tᵀ H T."""
        x_full = self.expand_to_full_dofs(self.scaling * x_reduced)
        H_full = self.full_hess(x_full, self.p)
        H_reduced = np.matmul(self.T.T, np.matmul(H_full, self.T))
        return H_reduced

    def update_precond(self, x_reduced):
        """Update preconditioner from reduced Hessian."""
        print("Updating with condensed Hessian preconditioner.")
        H_reduced = csc_matrix(self.hessian(x_reduced))
        self.precond.update(lambda attempt: H_reduced)

    def apply_precond(self, vx):
        return self.precond.apply(vx) if self.precond else vx

    def multiply_by_approx_hessian(self, vx):
        return self.precond.multiply_by_approximate(vx) if self.precond else vx

    def check_stability(self, x_reduced):
        if self.precond:
            self.precond.check_stability(x_reduced, self.p)
