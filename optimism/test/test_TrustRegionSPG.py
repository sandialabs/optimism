import numpy as onp
from jax.scipy import linalg
import unittest

from optimism.JaxConfig import *
from optimism import TrustRegionSPG
from optimism import Objective
from optimism.test import TestFixture

from scipy import sparse
from optimism import EquationSolver

def energy(x, params):
    p = params[0]
    return x[0]*(x[0]+1) + 0.3*x[1]*(x[1]-0.2) + 0.2*x[2]*(x[2]-0.5) + x[0]*x[0]*x[1]*x[1] + p*x[0]*x[1] + np.sin(x[0])


def quadratic(x, p):
    return p[0]*x[0]*x[0] + x[0]*x[1] + 3.0*x[1]*x[1] + x[2]*x[0] + 2.0*x[2]*x[2]


def rosenbrock(x, p):
    F = p[0]*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    F += p[0]*(x[2] - x[1]**2)**2 + (1 - x[1])**2
    return F


class TRSPGCauchyPointFixture(TestFixture.TestFixture):

    def setUp(self):
        self.settings = TrustRegionSPG.get_settings()
        self.mu0 = self.settings.cauchy_point_sufficient_decrease_factor

        # initial guess
        self.x = np.array([2., 7., -1.])

        self.p = Objective.Params(1.0)
        self.obj = Objective.Objective(energy, self.x, self.p)

    def test_projection(self):
        x = np.array([2.0, 2.0])
        lb = np.array([3.0, -np.inf])
        ub = np.array([np.inf, 1.0])
        bounds = np.column_stack((lb, ub))
        Px = TrustRegionSPG.project(x, bounds)
        self.assertTrue(np.all(Px >= lb))
        self.assertTrue(np.all(Px <= ub))

    def test_unconstrained_cauchy_point_achieves_sufficient_decrease(self):
        p = Objective.Params(1.0)
        x = np.array([3.0, 2.0])
        objective = Objective.Objective(quadratic, x, p)
        g = objective.gradient(x)
        lb = np.full(x.shape, -np.inf)
        ub = np.full(x.shape, np.inf)
        bounds = np.column_stack((lb,ub))
        alpha = 1.0
        trSize = 1e10 # huge, so it does not matter
        alpha, s = TrustRegionSPG.find_generalized_cauchy_point(x, g, lambda v: objective.hessian_vec(x, v), bounds, alpha, trSize, self.settings)
        print('xNew', x + alpha*s)
        qOld = quadratic(x, p)
        qNew = quadratic(x + alpha*s, p)
        self.assertLessEqual(qNew-qOld, self.mu0*s@g)

    def test_cauchy_point_obeys_constraints(self):
        p = Objective.Params(1.0)
        x = np.array([3.0, 2.0])
        objective = Objective.Objective(quadratic, x, p)
        g = objective.gradient(x)
        lb = np.array([-np.inf, 1.5])
        ub = np.full(x.shape, np.inf)
        bounds = np.column_stack((lb,ub))
        alpha = 1.0
        trSize = 1e10 # huge, so it does not matter
        alpha, s = TrustRegionSPG.find_generalized_cauchy_point(x, g, lambda v: objective.hessian_vec(x, v), bounds, alpha, trSize, self.settings)
        xCP = x + alpha*s
        self.assertTrue(np.all(xCP >= lb))
        self.assertTrue(np.all(xCP <= ub))

    def test_cauchy_point_with_active_constraints_achieves_sufficient_decrease(self):
        p = Objective.Params(1.0)
        x = np.array([3.0, 2.0])
        objective = Objective.Objective(quadratic, x, p)
        g = objective.gradient(x)
        lb = np.array([-np.inf, 1.5])
        ub = np.full(x.shape, np.inf)
        bounds = np.column_stack((lb,ub))
        alpha = 1.0
        trSize = 1e10 # huge, so it does not matter
        alpha, s = TrustRegionSPG.find_generalized_cauchy_point(x, g, lambda v: objective.hessian_vec(x, v), bounds, alpha, trSize, self.settings)
        qOld = quadratic(x, p)
        qNew = quadratic(x + alpha*s, p)
        self.assertLessEqual(qNew-qOld, self.mu0*s@g)

    def test_cauchy_point_stays_inside_trust_region(self):
        p = Objective.Params(1.0)
        x = np.array([3.0, 2.0])
        objective = Objective.Objective(quadratic, x, p)
        g = objective.gradient(x)
        lb = np.full(x.shape, -np.inf)
        ub = np.full(x.shape, np.inf)
        bounds = np.column_stack((lb,ub))
        alpha = 1.0
        trSize = 1.0e-2
        alpha, s = TrustRegionSPG.find_generalized_cauchy_point(x, g, lambda v: objective.hessian_vec(x, v), bounds, alpha, trSize, self.settings)
        print('xNew', x + alpha*s)
        qOld = quadratic(x, p)
        qNew = quadratic(x + alpha*s, p)
        sNorm = np.linalg.norm(s)
        print('step norm', sNorm)
        self.assertLessEqual(sNorm, trSize)

    def test_forward_track_search_of_cp_step_length_achieves_sufficient_decrease(self):
        p = Objective.Params(1.0)
        x = np.array([3.0, 2.0])
        objective = Objective.Objective(quadratic, x, p)
        g = objective.gradient(x)
        lb = np.full(x.shape, -np.inf)
        ub = np.full(x.shape, np.inf)
        bounds = np.column_stack((lb,ub))
        # pick small initial step guess so that forward track
        # will be used
        alpha = 1.0e-3
        # verify that forward track will be used
        s = TrustRegionSPG.project(x - alpha*g, bounds) - x
        ss = s@s
        model = 0.5*s@objective.hess_vec(x, p, s) + g@s
        trSize = 1e10 # huge, so it does not matter
        intialStepAcceptable = model <= self.mu0*g@s and ss <= trSize*trSize
        self.assertTrue(intialStepAcceptable)
        alpha, s = TrustRegionSPG.find_generalized_cauchy_point(x, g, lambda v: objective.hessian_vec(x, v), bounds, alpha, trSize, self.settings)
        qOld = quadratic(x, p)
        qNew = quadratic(x + alpha*s, p)
        #print('xNew', x + alpha*s)
        #print('qOld', qOld)
        #print('qNew', qNew)
        #print('minimal acceptable decrease', mu0*s@g)
        self.assertLessEqual(qNew-qOld, self.mu0*s@g)

    def no_test_trust_region_equation_solver(self):
        sol = es.trust_region_least_squares_solve(self.obj, self.x, self.settings)
        self.assertNear( np.linalg.norm( self.obj.gradient(sol)), 0, 7 )



class TestProjectionOnBoundary(TestFixture.TestFixture):

    def setUp(self):
        self.xk = np.zeros((3,))
        self.x = np.array([2., 7., -1.])
        self.lb = np.array([0.0, 0.0, 0.0])
        self.ub = np.array([5.0, 5.0, 5.0])
        self.bounds = np.column_stack((self.lb,self.ub))

    def test_project_onto_tr_function_when_vector_is_inside_tr(self):
        trSize = np.linalg.norm(self.x)*1.2
        xNew = TrustRegionSPG.project_onto_tr(self.x, self.xk, self.bounds, trSize)
        self.assertTrue(np.all(xNew >= self.lb))
        self.assertTrue(np.all(xNew <= self.ub))

    def test_project_onto_tr_function_when_vector_is_outside_tr(self):
        trSize = 4.0
        xNew = TrustRegionSPG.project_onto_tr(self.x, self.xk, self.bounds, trSize)
        self.assertTrue(np.all(xNew >= self.lb))
        self.assertTrue(np.all(xNew <= self.ub))
        self.assertLess(np.linalg.norm(xNew - self.xk), trSize*(1.0 + 1e-8))

        
class TestSubproblemSolveFixture(TestFixture.TestFixture):

    def setUp(self):
        self.settings = TrustRegionSPG.get_settings(spg_inexact_solve_ratio=1e-12, spg_tol=1e-4)
        # initial guess
        self.x = np.array([2., 7., -1.])
        self.p = Objective.Params(1.0)
        self.obj = Objective.Objective(quadratic, self.x, self.p)
        lb = np.array([1.0, -np.inf, -np.inf])
        ub = np.full(self.x.shape, np.inf)
        self.bounds = np.column_stack((lb,ub))
        self.g = self.obj.gradient(self.x)
        def Hv(v): return self.obj.hessian_vec(self.x, v)
        self.hess_vec = Hv
        
    def test_subproblem_solve_inside_tr(self):
        alpha = 1.0
        trSize = 1e10
        _, cp = TrustRegionSPG.find_generalized_cauchy_point(
            self.x, self.g, self.hess_vec, self.bounds, alpha,
            trSize, self.settings)
        # print('cp', cp)
        z, q, opt, stepType, i = TrustRegionSPG.solve_spg_subproblem(
            self.x, cp, self.g, self.bounds, self.hess_vec, None, trSize, self.settings)
        # print('iters taken', i)
        xNew = self.x + z
        xExact = np.array([1.0, -1/6, -1/4])
        self.assertArrayNear(xNew, xExact, 4)

        # unconstrained DOFS should have zero residual
        residual = self.obj.gradient(xNew)
        unconstrainedResidNorm = np.linalg.norm(residual[1:])
        self.assertLess(unconstrainedResidNorm, self.settings.spg_tol)

        # check constraint force of bounded DOF
        LagrangeMultiplierExact = -19/12
        self.assertAlmostEqual(residual[0], -LagrangeMultiplierExact, 4)

        # step should be inside trust region
        self.assertLessEqual(np.linalg.norm(z), trSize)
        
        #print('z', z)
        #print('norm z', np.linalg.norm(z))
        #print('trsize', trSize)
        #rint('residual', self.obj.gradient(xNew))
        #print('iters', i)

    def test_subproblem_solve_on_tr_boundary(self):
        alpha = 1.0
        # This is the full step that would be taken if the
        # trust region boundary was big enough
        xFullStep = np.array([1.0, -1/6, -1/4])
        trSize = 0.75*np.linalg.norm(xFullStep)
        _, cp = TrustRegionSPG.find_generalized_cauchy_point(
            self.x, self.g, self.hess_vec, self.bounds, alpha,
            trSize, self.settings)
        print('trSize', trSize)
        print('cp', cp)
        print('cp step size', np.linalg.norm(cp))
        z, q, opt, stepType, i = TrustRegionSPG.solve_spg_subproblem(
            self.x, cp, self.g, self.bounds, self.hess_vec, None, trSize, self.settings)
        xNew = self.x + z
        print('step size', np.linalg.norm(z))
        print('trSize', trSize)

        self.assertAlmostEqual(np.linalg.norm(z), trSize, 8)

        self.assertLess(q, quadratic(self.x, self.p))

        self.assertGreaterEqual(xNew[0], self.bounds[0,0])
        
        #print('iters', i)


class TestTrustRegionSPGFixture(TestFixture.TestFixture):

    def setUp(self):
        self.settings = TrustRegionSPG.get_settings()

        # initial guess
        self.x = np.array([2., 7., -1.])
        
        self.p = Objective.Params(1.0)
        self.obj = Objective.Objective(energy, self.x, self.p)
        
    def test_trust_region_spg_on_unbounded_problem(self):
        lb = np.full(self.x.shape, -np.inf)
        ub = np.full(self.x.shape, np.inf)
        sol = TrustRegionSPG.solve(self.obj, self.x, self.p, lb, ub,
                                   self.settings, useWarmStart=False)
        self.assertNear(np.linalg.norm(self.obj.gradient(sol)), 0, 7)

        # BT: This solver is converging to the same point as the
        # Steihaug solver in EquationSolver, which is good.
        # That point has a negative eigenvalue in the Hessian,
        # which is bad. I think this objective function may be
        # badly formed for this problem.
        # Otherwise, this might be a rare example that causes a
        # saddle point to be found. More study needed.
        # H = self.obj.hess(self.x, self.p)
        # eigvals = linalg.eigh(H, eigvals_only=True)
        # print('eigvals', eigvals)
        # self.assertTrue(onp.testing.assert_array_less(-eigvals, np.zeros(self.x.shape)))

    def no_test_cgunbound(self):
        settings = EquationSolver.get_settings()
        sol = EquationSolver.nonlinear_equation_solve(self.obj, self.x, self.p,
                                                      settings, useWarmStart=False)
        self.assertNear(np.linalg.norm(self.obj.gradient(sol)), 0, 7)


class TestTrustRegionSPGRosenbrock(TestFixture.TestFixture):
    
    def setUp(self):
        self.settings = TrustRegionSPG.get_settings(max_spg_iters=25,
                                                    cauchy_point_max_line_search_iters=10,
                                                    eta1=0.05,
                                                    eta2=0.05,
                                                    eta3=0.9,
                                                    t2=2.5)

        # initial guess
        self.x = np.array([-0.5, 8., -2.])
        self.p = Objective.Params(10.0)
        def id_precond(x, p):
            return sparse.identity(x.shape[0], format='csc')
        precondStrategy = Objective.PrecondStrategy(id_precond)
        self.obj = Objective.Objective(rosenbrock, self.x, self.p, precondStrategy)

    def no_test_hessian(self):
        H = self.obj.hess(self.x, self.p)
        print('H', H)
        eigvals = linalg.eigh(H, eigvals_only=True)
        print('eigvals', eigvals)
        
    def test_spg_on_rosenbrock(self):
        lb = np.full(self.x.shape, -np.inf)
        ub = np.full(self.x.shape, np.inf)
        sol = TrustRegionSPG.solve(self.obj, self.x, self.p, lb, ub,
                                   self.settings, useWarmStart=False)
        #print('sol', sol)
        self.assertNear(np.linalg.norm(self.obj.gradient(sol)), 0, 7)
        self.assertArrayNear(sol, np.ones(sol.shape), 4)


    def no_test_steihaug_on_rosenbrock(self):
        settings = EquationSolver.get_settings()
        sol = EquationSolver.nonlinear_equation_solve(self.obj, self.x, self.p,
                                                      settings, useWarmStart=False)
        print('sol', sol)
        
        
if __name__ == '__main__':
    unittest.main()
