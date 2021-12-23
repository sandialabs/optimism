from optimism.JaxConfig import *
from jax import custom_jvp

def zmax(x, eps):
    tmp = if_then_else(x >= eps, x, (x+eps)**2/(4.0*eps))
    return if_then_else(x <= -eps, 0.0, tmp)


def min(x, y, eps):
    xmy = x-y
    justMin = np.where(x <= y, x, y)
    # use double where to avoid nans in gradients
    isInsideEps = np.abs(xmy) < eps
    x = np.where(isInsideEps, x, 0.0)
    y = np.where(isInsideEps, y, 0.0)
    return np.where(isInsideEps, (-0.25*(x+y-eps)**2 + x*y)/eps, justMin)


safeTol=1e-14

def min_base(x, y, eps):
    safeEps = np.where(eps > safeTol, eps, safeTol)
    xmy = x-y
    justMin = np.where(x < y, x, y)
    isInsideEps = np.abs(xmy) < eps
    x = np.where(isInsideEps, x, 0.0)
    y = np.where(isInsideEps, y, 0.0)
    return np.where(isInsideEps, (-0.25*(x+y-safeEps)**2 + x*y)/safeEps, justMin)


def safe_min(x, y, eps):
    return min_base(x, y, eps)
