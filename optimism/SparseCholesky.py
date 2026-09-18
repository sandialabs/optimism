from scipy.sparse import isspmatrix_csc, isspmatrix_csr
from scipy.sparse import csc_matrix, identity

import numpy as onp

from sksparse import cholmod
from sksparse.cholmod import CholmodNotPositiveDefiniteError as NotPosDefError

from optimism.JaxConfig import *


class SparseCholesky:
    
    def factorize(self): 

        print('Assembling preconditioner', 0)
        stiffnessTryStep = 0
        self.A = self.new_stiffness_func(stiffnessTryStep)
        
        # doing the analyze every time for now, even if we know the sparsity does not change
        # we can improve this later if we are inclined
        assert isspmatrix_csc(self.A),  \
            "Preconditioner matrix is not in a valid sparse format"
        self.Precond = cholmod.CholeskyFactor(self.A, sym_kind="sym", supernodal_mode="supernodal",
                                              order="metis")

        attempt = 0
        maxAttempts = 10
        while attempt < maxAttempts:
            try:
                print('Factorizing preconditioner')
                self.Precond.factorize(self.A)
            except NotPosDefError:
                attempt += 1
                print('Cholesky failed, assembling preconditioner', attempt)
                # we are assuming that the sparsity does not change here
                self.A = self.new_stiffness_func(attempt)
            else:
                break
            
        if attempt == maxAttempts:
            print("Cholesky failed too many times, using identity preconditioner")
            self.A = identity(self.A.shape[0], format='csc')
            self.Precond.factorize(self.A)

            
    def update(self, new_stiffness_func):
        self.new_stiffness_func = new_stiffness_func
        self.factorize()

        
    def apply(self, b):
        return np.asarray(self.Precond.solve(onp.array(b, copy=True)))

        
    def apply_transpose(self, b):
        return self.apply(b)
   
    
    def multiply_by_approximate(self, x):
        return self.A.dot(x)
    
    
    def multiply_by_transpose(self, x):
        return self.A.T.dot(x)


    def get_diagonal_stiffness(self):
        return self.A.diagonal()

    
    def __matmul__(self, b):
        return self.A.dot(b)

