import matplotlib.pyplot as plt
from optimism.JaxConfig import *

N = 150
x = np.linspace(-0.05, 1.25, N)

def f(x):
    return -np.cos(7.14*x)*x + 0.2*x

y = f(x)
dy = vmap(grad(f))(x)
dydy = 0.5*dy*dy

plt.plot(x,15*y,'r',x,dydy,'b')

plt.savefig('sin.png')
