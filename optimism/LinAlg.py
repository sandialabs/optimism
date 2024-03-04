import jax
import jax.numpy as np

from optimism.JaxConfig import if_then_else
from optimism.QuadratureRule import create_padded_quadrature_rule_1D

@jax.custom_jvp
def sqrtm(A):
    sqrtA,_ = sqrtm_dbp(A)
    return sqrtA


@sqrtm.defjvp
def jvp_sqrtm(primals, tangents):
    A, = primals
    H, = tangents
    sqrtA = sqrtm(A)
    dim = A.shape[0]
    # TODO(brandon): Use a stable algorithm for solving a Sylvester equation.
    # See https://en.wikipedia.org/wiki/Bartels%E2%80%93Stewart_algorithm
    # The following will only reliably work for small matrices.
    I = np.identity(dim)
    M = np.kron(sqrtA.T, I) + np.kron(I, sqrtA)
    Hvec = H.T.ravel()
    return sqrtA, (np.linalg.solve(M, Hvec)).reshape((dim,dim)).T


def sqrtm_dbp(A):
    """ Matrix square root by product form of Denman-Beavers iteration.
    
    Translated from the Matrix Function Toolbox
    http://www.ma.man.ac.uk/~higham/mftoolbox
    Nicholas J. Higham, Functions of Matrices: Theory and Computation,
    SIAM, Philadelphia, PA, USA, 2008. ISBN 978-0-898716-46-7,
    """
    dim        = A.shape[0]
    tol        = 0.5 * np.sqrt(dim) * np.finfo(np.dtype("float64")).eps
    maxIters   = 32
    scaleTol   = 0.01

    def scaling(M):
        d  = np.abs(np.linalg.det(M))**(1.0/(2.0*dim))
        g = 1.0 / d
        return g
    
    def cond_f(loopData):
        _,_,error,k,_ = loopData
        p = np.array([k < maxIters, error > tol], dtype=bool)
        return np.all(p)
    
    def body_f(loopData):
        X, M, error, k, diff = loopData
        g = np.where(diff >= scaleTol,
                     scaling(M),
                     1.0)
        
        X *= g
        M *= g * g
        
        Y = X
        N = np.linalg.inv(M)
        I = np.identity(dim)
        X = 0.5 * X @ (I + N)
        M = 0.5 * (I + 0.5 * (M + N))
        error = np.linalg.norm(M - I, 'fro')
        diff  = np.linalg.norm(X - Y, 'fro') / np.linalg.norm(X, 'fro')
        k += 1
        return (X, M, error, k, diff)

    X0        = A
    M0        = A
    error0    = np.finfo(np.dtype("float64")).max
    k0        = 0
    diff0     = 2.0*scaleTol # want to force scaling on first iteration
    loopData0 = (X0, M0, error0, k0, diff0)
    
    X,_,_,k,_ = jax.lax.while_loop(cond_f, body_f, loopData0)

    return X,k


@jax.custom_jvp
def logm_iss(A):
    X,k,m = _logm_iss(A)
    return (1 << k) * log_pade_pf(X - np.identity(A.shape[0]), m)


@logm_iss.defjvp
def logm_jvp(primals, tangents):
    A, = primals
    H, = tangents
    logA = logm_iss(A)
    DexpLogA = jax.jacfwd(jax.scipy.linalg.expm)(logA)
    dim = A.shape[0]
    JVP = np.linalg.solve(DexpLogA.reshape(dim*dim,-1), H.ravel())
    return logA, JVP.reshape(dim,dim)


def _logm_iss(A):
    """Logarithmic map by inverse scaling and squaring and Padé approximants
    
    Translated from the Matrix Function Toolbox
    http://www.ma.man.ac.uk/~higham/mftoolbox
    Nicholas J. Higham, Functions of Matrices: Theory and Computation,
    SIAM, Philadelphia, PA, USA, 2008. ISBN 978-0-898716-46-7,
    """
    dim = A.shape[0]
    c15 = log_pade_coefficients[15]

    def cond_f(loopData):
        _,_,k,_,_,converged = loopData
        conditions = np.array([~converged, k < 16], dtype = bool)
        return conditions.all()

    def compute_pade_degree(diff, j, itk):
        j += 1
        # Manually force the return type of searchsorted to be 64-bit int, because it
        # returns 32-bit ints, ignoring the global `jax_enable_x64` flag. This looks
        # like a bug. I filed an issue (#11375) with Jax to correct this.
        # If they fix it, the conversions on p and q can be removed.
        p = np.searchsorted(log_pade_coefficients[2:16], diff, side='right').astype(np.int64)
        p += 2
        q = np.searchsorted(log_pade_coefficients[2:16], diff/2.0, side='right').astype(np.int64)
        q += 2
        m,j,converged = if_then_else((2 * (p - q) // 3 < itk) | (j == 2),
                                     (p+1,j,True), (0,j,False))
        return m,j,converged

    def body_f(loopData):
        X,j,k,m,itk,converged = loopData
        diff = np.linalg.norm(X - np.identity(dim), ord=1)
        m,j,converged = if_then_else(diff < c15,
                                     compute_pade_degree(diff, j, itk),
                                     (m, j, converged))
        X,itk = sqrtm_dbp(X)
        k += 1
        return X,j,k,m,itk,converged

    X   = A
    j   = 0
    k   = 0
    m   = 0
    itk = 5
    converged = False
    X,j,k,m,itk,converged = jax.lax.while_loop(cond_f, body_f, (X,j,k,m,itk,converged))
    return X,k,m


def log_pade_pf(A, n):
    """Logarithmic map by Padé approximant and partial fractions
    """
    I = np.identity(A.shape[0])
    X = np.zeros_like(A)
    quadPrec = 2*n - 1
    xs,ws = create_padded_quadrature_rule_1D(quadPrec)

    def get_log_inc(A, x, w):
        B = I + x*A
        dXT = w*np.linalg.solve(B.T, A.T)
        return dXT

    dXsTransposed = jax.vmap(get_log_inc, (None, 0, 0))(A, xs, ws)
    X = np.sum(dXsTransposed, axis=0).T
        
    return X


log_pade_coefficients = np.array([
    1.100343044625278e-05, 1.818617533662554e-03, 1.620628479501567e-02, 5.387353263138127e-02,
    1.135280226762866e-01, 1.866286061354130e-01, 2.642960831111435e-01, 3.402172331985299e-01,
    4.108235000556820e-01, 4.745521256007768e-01, 5.310667521178455e-01, 5.806887133441684e-01,
    6.240414344012918e-01, 6.618482563071411e-01, 6.948266172489354e-01, 7.236382701437292e-01,
    7.488702930926310e-01, 7.710320825151814e-01, 7.905600074925671e-01, 8.078252198050853e-01,
    8.231422814010787e-01, 8.367774696147783e-01, 8.489562661576765e-01, 8.598698723737197e-01,
    8.696807597657327e-01, 8.785273397512191e-01, 8.865278635527148e-01, 8.937836659824918e-01,
    9.003818585631236e-01, 9.063975647545747e-01, 9.118957765024351e-01, 9.169328985287867e-01,
    9.215580354375991e-01, 9.258140669835052e-01, 9.297385486977516e-01, 9.333644683151422e-01,
    9.367208829050256e-01, 9.398334570841484e-01, 9.427249190039424e-01, 9.454154478075423e-01,
    9.479230038146050e-01, 9.502636107090112e-01, 9.524515973891873e-01, 9.544998058228285e-01,
    9.564197701703862e-01, 9.582218715590143e-01, 9.599154721638511e-01, 9.615090316568806e-01,
    9.630102085912245e-01, 9.644259488813590e-01, 9.657625632018019e-01, 9.670257948457799e-01,
    9.682208793510226e-01, 9.693525970039069e-01, 9.704253191689650e-01, 9.714430492527785e-01,
    9.724094589950460e-01, 9.733279206814576e-01, 9.742015357899175e-01, 9.750331605111618e-01,
    9.758254285248543e-01, 9.765807713611383e-01, 9.773014366339591e-01, 9.779895043950849e-01 ])
