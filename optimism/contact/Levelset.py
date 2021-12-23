from optimism.JaxConfig import *


def plane(x, yLoc):
    return yLoc-x[:,1]

# 5,5 should be positive
def corner(x, xLoc, yLoc):
    return np.minimum( x[:,0]-xLoc, x[:,1]-yLoc )


def sphere(x, xLoc, yLoc, R):
    r = np.sqrt( (x[:,0]-xLoc)**2 + (x[:,1]-yLoc)**2 ) - R
    return r


def combined(x, ls1, ls2):
    return np.minimum( ls1(x), ls2(x) )
