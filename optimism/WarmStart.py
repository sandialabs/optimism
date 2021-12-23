from scipy.sparse.linalg import cg, LinearOperator

from optimism.JaxConfig import *
import optimism.Objective as Objective
    
def warm_start_increment(objective, x, pNew, index=0):
    dp = objective.p[index] - pNew[index]

    if index==0:
        b = objective.jacobian_p_vec(x, dp)
    elif index==2:
        b = objective.jacobian_p2_vec(x, dp)
    else:
        raise('invalid warm start parameter gradient direction')
        
    sz = b.size
    op = lambda v: objective.hessian_vec(x, v)
    
    Lop = LinearOperator((sz,sz),
                         matvec = op)
    LopPrecond = LinearOperator((sz,sz),
                                matvec = objective.apply_precond)
    
    numIters = 0
    def callback(xk):
        nonlocal numIters
        numIters += 1

    # residual = L dx - b

    dx, cgWarmStartSolveSuccess = cg(Lop, b, M=LopPrecond, callback=callback)
    print('num warm start cg iters = ', numIters)
    # assert(cgWarmStartSolveSuccess==0)
    
    return dx 


from jax import device_put

def warm_start_increment_jax_safe(objective, x, pNew):
    dp0 = objective.p[0] - pNew
    b = objective.jacobian_p_vec(x, dp0)
    
    try:
        b = b.primal
        op = lambda v: objective.hessian_vec(x, v).primal
    except AttributeError:
        op = lambda v: objective.hessian_vec(x, v)

    sz = b.shape[0]
    Lop = LinearOperator((sz,sz),
                         matvec = op)
    LopPrecond = LinearOperator((sz,sz),
                                matvec = objective.apply_precond)
    
    numIters = 0
    def callback(xk):
        nonlocal numIters
        numIters += 1

    # residual = L dx - b

    dx, cgWarmStartSolveSuccess = cg(Lop, b, M=LopPrecond, callback=callback)
    print('num warm start cg iters = ', numIters)
    # assert(cgWarmStartSolveSuccess==0)
    
    return dx 
