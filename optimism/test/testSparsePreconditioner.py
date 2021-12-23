import scipy.sparse
from scipy.sparse import identity
import numpy as onp
import sys

from optimism.JaxConfig import *
from optimism.test.TestFixture import *
try:
    from optimism.SparseCholesky import SparseCholesky
except ImportError:
    print('Sparse Cholesky solvers not installed')
    print('Skipping all asociated tests')

    
haveSparseCholesky = 'optimism.SparseCholesky' in sys.modules
skipMessage = 'Sparse Cholesky solvers not installed'


class SparseCholeskyFixture(TestFixture):

    def setUp(self):
        self.A = scipy.sparse.diags([[2,2,2,2,2,1], -1*np.ones(5), -1*np.ones(5)],[0,-1,1],shape=(6,6),format='csc', dtype=float)
        self.stiffness_func = lambda x, p: p*self.A
        self.b = np.array([0, 0, 0, 0, 0, 1])
        self.P = SparseCholesky()


    def precond_update_func(self, matrixScale):
        return lambda attempt: self.stiffness_func(None, matrixScale) + attempt*identity(self.b.shape[0])
        
    @unittest.skipIf(not haveSparseCholesky, skipMessage)
    def test_sparse_solve(self):
        matrixScale = 1.0
        self.P.update( self.precond_update_func(matrixScale) )
        x = self.P.apply(self.b)
        self.assertArrayNear(self.b, self.A.dot(x), 14)

        
    @unittest.skipIf(not haveSparseCholesky, skipMessage)
    def test_sparse_solve_and_update(self):
        
        matrixScale = 1.0
        self.P.update( self.precond_update_func(matrixScale) )
        x1 = self.P.apply(self.b)
        
        matrixScale = 2.0
        self.P.update( self.precond_update_func(matrixScale) )
        x2 = self.P.apply(self.b)
        
        self.assertArrayNotEqual(x1, x2)
        self.assertArrayNear(self.b, matrixScale*self.A.dot(x2), 14)

        
    @unittest.skipIf(not haveSparseCholesky, skipMessage)
    def test_indefinite_fixed_by_shift(self):
        
        matrixScale = 1.0
        self.P.update( self.precond_update_func(matrixScale) )
        x1 = self.P.apply(self.b)

        matrixScale = -1.0
        self.P.update( self.precond_update_func(matrixScale) )
        x2 = self.P.apply(self.b)

        # no exception thrown because shift made it invertible
        # approx solve is different
        self.assertArrayNotEqual(x1, x2)


    @unittest.skipIf(not haveSparseCholesky, skipMessage)
    def test_multiply_by_transpose(self):
        matrixScale = 2.0
        self.P.update( self.precond_update_func(matrixScale) )
        
        x = np.arange(self.A.shape[1], dtype=self.A.dtype)
        b = self.P.multiply_by_transpose(x)
        self.assertArrayEqual(b, matrixScale*self.A.T.dot(x))


    @unittest.skipIf(not haveSparseCholesky, skipMessage)
    def test_diagonal_backup_preconditioner(self):
        M = scipy.sparse.csc_matrix(onp.array([[1.0, -1e12, -1e12],
                                               [-1e12, 1.0,  -1e12],
                                               [-1e12, -1e12, 1.0]]))
        self.P.update(lambda _: M)
        self.assertArrayEqual(self.P.A.todense(), onp.identity(3))
            

if __name__ == '__main__':
    unittest.main()
