from optimism.JaxConfig import *
from optimism import MinimizeScalar
from optimism.test import TestFixture

from optimism.material import J2Plastic

def f(x): return 0.25*x**4 - 50.0*x**2 + 2.0
df = jacfwd(f)

class TestMinimizeScalarFixture(TestFixture.TestFixture):

    def setUp(self):
        self.minimize_scalar_jitted = jit(MinimizeScalar.minimize_scalar, static_argnums=(0,4))


    def test_solves_quadratic_problem_in_one_iteration(self):
        f = lambda x: x*x
        x0 = 3.5
        settings = MinimizeScalar.get_settings(tol=1e-8, max_iters=1)
        x = MinimizeScalar.minimize_scalar(f, x0,
                                           diffArgs=tuple(), nondiffArgs=tuple(),
                                           settings=settings)
        self.assertNear(x, 0.0, 12)

        
    def test_does_not_converge_to_saddle_point(self):
        x0 = -0.001
        settings = MinimizeScalar.get_settings(tol=1e-10, max_iters=30)
        x = MinimizeScalar.minimize_scalar(f, x0,
                                           diffArgs=tuple(), nondiffArgs=tuple(),
                                           settings=settings)
        r = np.abs(df(x))
        self.assertLess(r, settings.tol)
        
        self.assertNear(x, -10.0, 9)


    def notest_jit(self):
        x0 = -0.001
        settings = MinimizeScalar.get_settings(tol=1e-10, max_iters=30)
        x = self.minimize_scalar_jitted(f, x0,
                                        diffArgs=tuple(), nondiffArgs=tuple(),
                                        settings=settings)
        print("x={:1.13e}".format(x))
        self.assertNear(x, -1.0, 9)
        

    def notest_grad(self):
        def g(x,c): return 0.25*x**4 - 0.5*(c*x)**2 + 2.0
        c = -2.0
        x0 = -3.0
        settings = MinimizeScalar.get_settings(tol=1e-10, max_iters=30)
        x = MinimizeScalar.minimize_scalar(g, x0,
                                           diffArgs=(c,), nondiffArgs=tuple(),
                                           settings=settings)
        print("x={:1.13e}".format(x))
        self.assertNear(x, c, 10)


    def notest_stiff_problem(self):
        E = 69.0
        Y0 = 350.0
        n = 3.0
        eps0 = 1.0
        e = 1.01*Y0/E
        def Wp(ep):
            w = np.where(ep > 0.0,
                         Y0*ep + Y0*eps0*n/(n + 1.0)*(ep/eps0)**(1+1/n),
                         Y0*ep)
            return w
                         
        W = lambda ep: 0.5*E*(e - ep)**2 + Wp(ep)
        settings = MinimizeScalar.get_settings(tol=1e-8*Y0, max_iters=30)
        ep = MinimizeScalar.minimize_scalar(W, 1e-15, diffArgs=tuple(), nondiffArgs=tuple(),
                                            settings=settings)
        print("ep = ", ep)
        yield_func = grad(W)
        print("r=", -yield_func(ep))

        
if __name__ == '__main__':
    TestFixture.unittest.main()
