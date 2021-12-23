from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import isspmatrix_csc
import numpy as onp
import ilupp

from optimism.JaxConfig import *


class IncompleteCholesky:

    def update(self, hessian_func):
        print('updating preconditioner')
        precondAttempt = 0
        self.A = hessian_func(precondAttempt)
        assert isspmatrix_csc(self.A),  \
            "Preconditioner matrix is not in a valid sparse format"
        self.ordering = reverse_cuthill_mckee(self.A, symmetric_mode=True)
        self.invOrdering = onp.argsort(self.ordering)

        print('factorizing preconditioner')
        self.Precond = ilupp.IChol0Preconditioner(self.A[self.ordering][:,self.ordering])
        print('factorization complete')
        

    def apply(self, b):
        bHat = onp.array(b[self.ordering])
        return self.Precond.dot(bHat)[self.invOrdering]

    
    def multiply_by_approximate(self, x):
        return self.A.dot(x)


    def __matmul__(self, x):
        return self.A.dot(x)
