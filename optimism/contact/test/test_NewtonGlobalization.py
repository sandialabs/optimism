from numpy.random import rand

from optimism.JaxConfig import *
from optimism.test.TestFixture import *
from optimism.NewtonSolver import *
from optimism import AlSolver
from optimism.Objective import Objective
from optimism import EquationSolver as EqSolver
from optimism.ConstrainedObjective import ConstrainedObjective


class TestQuadraticSolver(TestFixture):

    def setUp(self):
        self.bounds = np.array([0.05, 0.75])
        self.samplePoints = np.linspace( self.bounds[0], self.bounds[1], 200 )

        
    def check_quadratic(self, ps, theta):
        p = construct_quadratic(ps)
        self.assertNear(p(0), ps[0], 14)
        self.assertNear(p(1), ps[1], 14)

        pmin = min(p(self.samplePoints))
        self.assertTrue( p(theta) <= pmin )

        
    def test_constant(self):
        ps = np.array([0.5, 0.5, 0.])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, self.bounds[1], 13)
        self.check_quadratic(ps, theta)

        
    def test_linear(self):
        ps = np.array([0.5, 0.9, 0.4])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, self.bounds[0], 13)
        self.check_quadratic(ps, theta)

        
    def test_negative_linear(self):
        ps = np.array([0.5, 0.1, -0.4])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, self.bounds[1], 13)
        self.check_quadratic(ps, theta)

        
    def test_negative_curvature(self):
        ps = np.array([0.5, 0.1, 0.0])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, self.bounds[1], 13)
        self.check_quadratic(ps, theta)

        
    def test_positive_curvature(self):
        ps = np.array([0.25, 0.25, -1.])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, 0.5, 13)
        self.check_quadratic(ps, theta)

        
    def test_positive_curvature2(self):
        ps = np.array([0.5, 2.5, 1.])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, self.bounds[0], 13)
        self.check_quadratic(ps, theta)

        
    def test_positive_curvature3(self):
        ps = np.array([2.5, 0.5, -2.5])
        theta = compute_min_p(ps, self.bounds)
        self.assertNear(theta, self.bounds[1], 13)
        self.check_quadratic(ps, theta)
    

def objective(x):
    return x[0]*(x[0]+1.0) + 0.3*x[1]*(x[1]-0.2) + 0.2*x[2]*(x[2]-0.5) + x[3]*x[3] + 0.00*(x[0]*x[0]*x[1]*x[1] + x[0]*x[1])


def constraint(x):
    return np.array([x@x-1.0, x[0]+x[1]-0.5, 1.0-x[0]])


sizex=4
sizelam=3


def fischer_burmeister(c, l):
    return np.sqrt(c**2+l**2) - c - l


dObjective = grad(objective)
dConstraint = grad(lambda x,lam: lam @ constraint(x))


def residual(xlam):
    x = xlam[:sizex]
    lam = xlam[sizex:]
    c = constraint(x)
    return np.hstack( (dObjective(x) - dConstraint(x, lam), fischer_burmeister(c, lam)) )


def create_linear_op(residual):
    return lambda x,v: jvp( residual , (x,), (v,) )[1]


@jit
def linear_op(xlam, v):
    dr_func = create_linear_op(residual)
    return dr_func(xlam, v)


def my_func(x):
    return 3.*x**2 + 2.*x + 3.


class TestGMRESSolver(TestFixture):

    def setUp(self):
        self.x = np.ones(sizex)
        self.lam = np.ones(sizelam)

        self.etak=1e-4
        self.t=1e-4

        tol = 5e-9
        self.subProblemSettings = EqSolver.get_settings()        
        self.alSettings = AlSolver.get_settings()

        
    def test_newton_step(self):
        xl = np.hstack( (self.x, self.lam) )
        r0 = np.linalg.norm( residual(xl) )
        
        xl += newton_step(residual, lambda v: linear_op(xl,v), xl)[0]
        r1 = np.linalg.norm( residual(xl) )

        self.assertTrue( r1 < 10*r0 )

        
    def test_globalized_newton_step_with_cubic(self):
        residual = vmap(lambda x: np.sqrt(my_func(x)))
        linear_op = create_linear_op(residual)
        x = np.array([0.1])
        r0 = np.linalg.norm( residual(x) )
        x += globalized_newton_step(residual, lambda v: linear_op(x,v), x, self.etak, self.t)
        r1 = np.linalg.norm( residual(x) )

        self.assertTrue( r1 < r0 )

        
    def test_globalized_newton_step_nonconvex(self):
        xl = np.hstack( (self.x, self.lam) )
        r0 = np.linalg.norm( residual(xl) )
        
        xl += globalized_newton_step(residual, lambda v: linear_op(xl,v), xl, self.etak, self.t)
        r1 = np.linalg.norm( residual(xl) )

        self.assertTrue( r1 < r0 )

        for count in range(2):
            xl += globalized_newton_step(residual, lambda v: linear_op(xl,v), xl, self.etak, self.t)
            r1 = np.linalg.norm( residual(xl) )
            
        
    def test_al_solver(self):
        # Guess a bit randomly at a good initial penalty stiffness
        p = None
        unconstrainedObjective = Objective(lambda x,p: objective(x), self.x, p)
        randRhs = np.array(rand(self.x.size))
        randRhs *= 0.25 / np.linalg.norm(randRhs)

        penalty0 = unconstrainedObjective.hessian_vec(self.x, randRhs) @ randRhs
        
        alObjective = ConstrainedObjective(lambda x,p: objective(x),
                                           lambda x,p: constraint(x),
                                           self.x,
                                           p,
                                           self.lam,
                                           penalty0 * np.ones(self.lam.shape))
        
        AlSolver.augmented_lagrange_solve(alObjective, self.x, p,
                                          self.alSettings, self.subProblemSettings, useWarmStart=False)
            
if __name__ == '__main__':
    unittest.main()
