from sys import float_info

import jax
import jax.numpy as np
from scipy.optimize import minimize_scalar, root_scalar # for comparison
from scipy.optimize import OptimizeResult

from optimism import ScalarRootFind
from optimism.test import TestFixture


def f(x): return x**3 - 4.0


class ScalarRootFindTestFixture(TestFixture.TestFixture):

    def setUp(self):
        self.settings = ScalarRootFind.get_settings()

        self.rootGuess = 1e-5
        self.rootExpected = np.cbrt(4.0)


        # this shows that a scipy root finder takes longer
        # sp_opts = {'xtol': self.settings.x_tol, 'maxiter': self.settings.max_iters}
        # result = root_scalar(f, method='brentq', bracket=self.rootBracket)
        # self.scipy_function_calls = result.function_calls
        # self.scipy_iterations = result.iterations
        # print('scipy root ', result.root)
        # print('scipy fevals ', result.function_calls)
        # print('scipy iterations ', result.iterations)
        

    def test_find_root(self):
        rootBracket = np.array([float_info.epsilon, 100.0])
        root, status = ScalarRootFind.find_root(f, self.rootGuess, rootBracket, self.settings)
        self.assertTrue(status.converged)
        self.assertNear(root, self.rootExpected, 13)

        
    def test_find_root_with_jit(self):
        rtsafe_jit = jax.jit(ScalarRootFind.find_root, static_argnums=(0,3))
        rootBracket = np.array([float_info.epsilon, 100.0])
        root, status = rtsafe_jit(f, self.rootGuess, rootBracket, self.settings)
        self.assertTrue(status.converged)
        self.assertNear(root, self.rootExpected, 13)


    def test_unbracketed_root_gives_nan(self):
        rootBracket = np.array([2.0, 100.0])
        root, _ = ScalarRootFind.find_root(f, self.rootGuess, rootBracket, self.settings)
        self.assertTrue(np.isnan(root))

        
    def test_find_root_converges_with_terrible_guess(self):
        rootBracket = np.array([float_info.epsilon, 200.0])
        root, status = ScalarRootFind.find_root(f, 199.0, rootBracket, self.settings)
        self.assertTrue(status.converged)
        self.assertNear(root, self.rootExpected, 13)

        
    def test_root_find_is_differentiable(self):
        myfunc = lambda x, a: x**3 - a
        def cube_root(a):
            rootBracket = np.array([float_info.epsilon, 100.0])
            root, _ = ScalarRootFind.find_root(lambda x: myfunc(x, a), 8.0, rootBracket, 
                                               self.settings)
            return root
                                            
        root = cube_root(4.0)
        self.assertNear(root, self.rootExpected, 13)

        df = jax.jacfwd(cube_root)
        x = 3.0
        self.assertNear(df(x), x**(-2/3)/3, 13)


    def test_find_root_with_forced_bisection_step(self):
        myfunc = lambda x, a: x**2 - a
        def my_sqrt(a):
            rootBracket = np.array([float_info.epsilon, 100.0])
            root, _ = ScalarRootFind.find_root(lambda x: myfunc(x, a), 8.0,
                                               rootBracket, self.settings)
            return root

        r = my_sqrt(9.0)
        self.assertNear(r, 3.0, 12)


    def test_root_find_with_vmap_and_jit(self):
        myfunc = lambda x, a: x**2 - a
        def my_sqrt(a):
            rootBracket = np.array([float_info.epsilon, 100.0])
            root, _ = ScalarRootFind.find_root(lambda x: myfunc(x, a), 8.0,
                                               rootBracket, self.settings)
            return root

        x = np.array([1.0, 4.0, 9.0, 16.0])
        F = jax.jit(jax.vmap(my_sqrt, 0))
        expectedRoots = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertArrayNear(F(x), expectedRoots, 12)


    def test_solves_when_left_bracket_is_solution(self):
        rootBracket = np.array([0.0, 1.0])
        guess = 3.0
        root, status = ScalarRootFind.find_root(lambda x: x*(x**2 - 10.0), guess, rootBracket, self.settings)
        self.assertTrue(status.converged)
        self.assertNear(root, 0.0, 12)


    def test_solves_when_right_bracket_is_solution(self):
        rootBracket = np.array([-1.0, 0.0])
        guess = 3.0
        root, status = ScalarRootFind.find_root(lambda x: x*(x**2 - 10.0), guess, rootBracket, self.settings)
        self.assertTrue(status.converged)
        self.assertNear(root, 0.0, 12)

if __name__ == '__main__':
    TestFixture.unittest.main()
