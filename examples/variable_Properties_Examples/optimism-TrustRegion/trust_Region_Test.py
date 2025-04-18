import numpy as np
import scipy as scp
import jax



func = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2
F = func((jax.numpy.array([-4. , 0.])))
dF = jax.grad(func)(jax.numpy.array([-4. , 0.]))
ddF = jax.hessian(func)(jax.numpy.array([-4. , 0.]))

ans = scp.optimize.minimize(func,[-4.0, 0.0],method = 'trust-constr')

print(ans)

