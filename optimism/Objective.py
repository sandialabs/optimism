from optimism.JaxConfig import *
from optimism.SparseCholesky import SparseCholesky
import numpy as onp
from scipy.sparse import diags as sparse_diags
from scipy.sparse import csc_matrix

# static vs dynamics
# differentiable vs undifferentiable
Params = namedtuple('Params',
                    ['bc_data',
                     'state_data',
                     'design_data',
                     'app_data',
                     'time',
                     'dynamic_data'],
                     defaults=(None,None,None,None,None,None))


def param_index_update(p, index, newParam):
    if index==0:
        return Params(newParam, p[1], p[2], p[3], p[4], p[5])
    if index==1:
        return Params(p[0], newParam, p[2], p[3], p[4], p[5])
    if index==2:
        return Params(p[0], p[1], newParam, p[3], p[4], p[5])
    if index==3:
        return Params(p[0], p[1], p[2], newParam, p[4], p[5])
    if index==4:
        return Params(p[0], p[1], p[2], p[3], newParam, p[5])
    if index==5:
        return Params(p[0], p[1], p[2], p[3], p[4], newParam)
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


class ObjectiveMPC(Objective):
    def __init__(self, f, x, p, dofManagerMPC, precondStrategy=None):
        """
        Initialize the Objective class for MPC with full DOFs and master-slave constraints.

        Parameters:
        - f: Objective function.
        - x: Initial guess for the primal DOF vector.
        - p: Parameters for the objective function.
        - dofManagerMPC: Instance of DofManagerMPC with master-slave relationships.
        - precondStrategy: Preconditioning strategy (optional).
        """
        super().__init__(f, x, p, precondStrategy)
        self.dofManagerMPC = dofManagerMPC
        self.master_dofs = dofManagerMPC.master_array[:, 1:]
        self.slave_dofs = dofManagerMPC.slave_array[:, 1:]
        self.slave_to_master_map = self.create_slave_to_master_map()

    def create_slave_to_master_map(self):
        """
        Create the mapping to express slave DOFs in terms of master DOFs.
        Returns:
        - C: Constraint matrix (2D array).
        - d: Constant vector (1D array).
        """
        num_slaves = len(self.slave_dofs)
        num_masters = len(self.master_dofs)
        
        # Initialize the constraint matrix with the correct dimensions
        C = np.zeros((num_slaves, num_masters))  # Replace with actual mapping logic
        
        # Initialize the constant vector
        d = np.zeros(num_slaves)  # Replace with actual constant values if needed
        
        return C, d


    def value(self, x_full):
        # Ensure the constraints are satisfied
        x_full = self.enforce_constraints(x_full)
        return super().value(x_full)

    def gradient(self, x_full):
        x_full = self.enforce_constraints(x_full)
        grad_full = super().gradient(x_full)
        return grad_full

    def hessian_vec(self, x_full, vx_full):
        x_full = self.enforce_constraints(x_full)
        hess_vec_full = super().hessian_vec(x_full, vx_full)
        return hess_vec_full

    def enforce_constraints(self, x_full):
        if self.slave_to_master_map is not None:
            C, d = self.slave_to_master_map
            slave_values = C @ x_full[self.master_dofs] + d[:, None]
            # print("Slave values before assignment:", slave_values)
            x_full = x_full.at[self.slave_dofs].set(slave_values)
            # print("x_full after enforcing constraints:", x_full)
        return x_full



