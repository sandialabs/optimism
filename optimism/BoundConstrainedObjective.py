from scipy.sparse import csc_matrix
from scipy.sparse import diags as sparse_diags
import numpy as onp

from optimism import ConstrainedObjective
from optimism.JaxConfig import *


class ScaledPrecondStrategy(ConstrainedObjective.PrecondStrategy):

    def __init__(self,
                 precondStrategy,
                 scaled_constraint_hessian,
                 dofScaling,
                 constrainedIndices):
        self.ps = precondStrategy
        self.constraint_hessian = scaled_constraint_hessian
        self.diagScaling = sparse_diags(onp.array(dofScaling), format='csc')
        self.constrainedIndices = constrainedIndices

        
    def initialize(self, x, p, lam, kappa):
        self.ps.initialize(self.diagScaling*x, p)
        self.constraintHessian = self.constraint_hessian(x, p, lam, kappa)

        
    def precond_at_attempt(self, attempt):
        K = self.ps.precond_at_attempt(attempt)
        K2 = csc_matrix( self.diagScaling.T * K * self.diagScaling ) + self.constraintHessian
        
        Kdiag = np.array(K2.diagonal().squeeze().ravel())
        diagConstrained = Kdiag[self.constrainedIndices]
        print('min, max constrained stiffness = ',
              np.min(diagConstrained),
              np.max(diagConstrained))
        
        print('min, max total stiffness = ',
              np.min(Kdiag),
              np.max(Kdiag))
        return K2


class BoundConstrainedObjective(ConstrainedObjective.ConstrainedObjective):
    
    def __init__(self,
                 objective_func,
                 x0,
                 p,
                 constrainedIndices,
                 constraintStiffnessScaling = 1.0,
                 precondStrategy=None):

        self.constrainedIndices = constrainedIndices
        
        if precondStrategy:
            precondStrategy.initialize(x0, p)
            K0 = precondStrategy.precond_at_attempt(0)
            scaling = np.sqrt(K0.diagonal())
            scaling = scaling.at[constrainedIndices].multiply(1.0/constraintStiffnessScaling)
            invScaling = 1.0/scaling


            def scaled_constraint_hessian(xBar, p, lam, kappa):
                c = xBar[constrainedIndices]
                phasePenaltyStiffness = np.where(lam >= c*kappa, kappa, 0.0)
                d = np.zeros_like(xBar).at[constrainedIndices].set(phasePenaltyStiffness)
                return sparse_diags(onp.array(d), format='csc')


            scaledPrecondStrategy = ScaledPrecondStrategy(precondStrategy, scaled_constraint_hessian, invScaling, constrainedIndices)

        else:
            scaling = np.ones_like(x0)
            invScaling = np.ones_like(x0)
            scaledPrecondStrategy = None

            
        lam0 = (grad(objective_func,0)(x0, p)*invScaling)[constrainedIndices]
        lam0 = np.maximum(lam0, 0.0)
        kappa0 = 0.25 * np.ones_like(lam0)

        
        def scaled_objective(xBar, p):
            x = invScaling * xBar
            return objective_func(x, p)


        def scaled_constraint_func(xBar, p):
            return xBar[constrainedIndices]


        xBar0 = scaling * x0
        super().__init__(scaled_objective,
                         scaled_constraint_func,
                         xBar0,
                         p,
                         lam0,
                         kappa0,
                         scaledPrecondStrategy)

        self.scaling = scaling
        self.invScaling = invScaling
        
        
    def get_multipliers(self):
        return self.lam * self.scaling[self.constrainedIndices]


    def get_value(self, x):
        return self.value(self.scaling * x)
    

    def get_residual(self, x):
        return self.gradient(self.scaling * x)

    
    def get_total_residual(self, x):
        return self.total_residual(self.scaling * x)
