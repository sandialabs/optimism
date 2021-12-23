from optimism.JaxConfig import *
from optimism import EquationSolver as es
from optimism import Objective
from optimism.test.TestFixture import TestFixture
import unittest

def energy(x, params):
    p = params[0]
    return x[0]*(x[0]+1) + 0.3*x[1]*(x[1]-0.2) + 0.2*x[2]*(x[2]-0.5) + x[0]*x[0]*x[1]*x[1] + p*x[0]*x[1] + np.sin(x[0])


class EquationSolverFixture(TestFixture):

    def setUp(self):
        self.settings = es.get_settings()

        # initial guess
        self.x = np.array([2., 7., -1.])
        self.p = Objective.Params(1.0)

        self.obj = Objective.Objective(energy, self.x, self.p)
        
    def test_trust_region_equation_solver(self):
        sol = es.trust_region_least_squares_solve(self.obj, self.x, self.settings)
        self.assertNear( np.linalg.norm( self.obj.gradient(sol)), 0, 7 )

    def test_trust_region_optimizer(self):
        sol = es.nonlinear_equation_solve(self.obj, self.x, self.p, self.settings)
        self.assertNear( np.linalg.norm( self.obj.gradient(sol)), 0, 7 )

    def test_trust_region_optimizer_with_preconditioned_inner_products(self):
        settings=es.get_settings(use_preconditioned_inner_product_for_cg=True)
        sol = es.nonlinear_equation_solve(self.obj, self.x, self.p, settings)
        self.assertNear( np.linalg.norm( self.obj.gradient(sol)), 0, 7 )

    def test_trust_region_incremental_optimizer(self):
        incrementalSettings = es.get_settings(use_incremental_objective=True)
        sol = es.nonlinear_equation_solve(self.obj, self.x, self.p, incrementalSettings)
        self.assertNear( np.linalg.norm( self.obj.gradient(sol)), 0, 7 )

if __name__ == '__main__':
    unittest.main()
