from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import cho_factor, cho_solve

# BT: This overwrites jax.numpy as the np namespace - is this the intent or a bug?
import numpy as np

class LU:
    
    def __init__(self, A):
        self.A = A
        try:
            self.Precond = lu_factor(self.A)
        except:
            print("Lu failed, using a diagonal preconditioner.")
            d = np.diag(self.A)
            eps = 1e-3 * d.mean()
            d = 0.*np.maximum(d, eps)+1.
            self.Precond = 1.0 / d


    
    def update(self, A):
        self.A = A
        try:
            print('Computing preconditioner')
            self.Precond = lu_factor(self.A)
        except:
            print("LU failed, not updating preconditioner.")
    
            
    def solve(self, b, transpose=0):
        if type(b) == type(np.array([])):
            b = np.array(b)
    
        if type(self.Precond) == type(np.array([])):
            return self.Precond*b
        else:
            return lu_solve(self.Precond, b, trans=transpose)
    
    
    def solve_transpose(self, b):
        return self.solve(b, 1)


    def dot(self, b):
        return self.A@b
    

    def multiply_by_transpose(self, b):
        return self.A.T@b


    def __matmul__(self, b):
        return self.A@b

