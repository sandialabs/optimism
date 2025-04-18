# from collections import namedtuple
# from jax import grad
# from jax import jit
# from optimism import EquationSolver
# from plato_optimism import exodus_writer as ExodusWriter
# from plato_optimism import GradUtilities
# from optimism import FunctionSpace
# from optimism import Interpolants
# from optimism import Mechanics
# from optimism import Mesh
# from optimism import Objective
# from optimism import QuadratureRule
# from optimism import ReadExodusMesh
# from optimism import SparseMatrixAssembler
# from optimism.FunctionSpace import DofManager
# from optimism.FunctionSpace import EssentialBC
# from optimism.inverse import MechanicsInverse
# from optimism.inverse import AdjointFunctionSpace
# from optimism.material import Neohookean_VariableProps
# import jax.numpy as np
# import numpy as onp
# from scipy.sparse import linalg
# from typing import Callable, NamedTuple

import exodus3 as exodus
import numpy as np
import os
import matplotlib.pyplot as plt

nx, ny = (20, 20)
K = 0.01 # cm^2 / (mW * s) <the conversion of cm^2 to m^2 and from mW to W cancel out
t = np.linspace(1, 10, nx) # seconds
I = np.linspace(20, 200, ny) # W/m^2

xv, yv = np.meshgrid(t, I)

p = 1 - np.exp(-K*xv*yv)

constProps = {'Ec': 1.059, # MPa
              'b': 5.248, # unitless
              'p_gel': 0.12, # unitless
              'Ed': 3.321, 
              'Er': 18959, 
              'R': 8.314, 
              'g1': 109603, # unitless
              'g2': 722.2, # unitless
              'xi': 3.73,# unitless
              'C1': 61000, # unitless
              'C2': 511.792, # K
              'rmTemp': 100, # C
              'tau': 0.001 # s
              }

# Ec = 1.059 # MPa
# b = 5.248 # unitless
# pgel = 0.12 # unitless
# Ed = 3.321 # MPa 


# E = (Ec * np.exp(b * (p - pgel))) + Ed

E = (constProps['Ec'] * np.exp(constProps['b'] * (p - constProps['p_gel']))) + constProps['Ed']
print(E)
glass = constProps['Er']/(constProps['R'] * np.log((constProps['g1'] * ((1 - p)**constProps['xi'])) + constProps['g2']))
glassC = glass - 273.15

# a = (-constProps['C1']*((constProps['rmTemp']+273.15) - glass))/(constProps['C2'] + ((constProps['rmTemp']+273.15) - glass))
# a = (constProps['C1']/(constProps['rmTemp'] - (glass - constProps['C2']))) - (constProps['C1']/(glass - (glass - constProps['C2']))) 
# a = (constProps['C1']/(constProps['rmTemp'] - glass + constProps['C2'])) - (constProps['C1']/constProps['C2'])
# ath = 10**(a)
# relaxTime = constProps['tau'] * ath


h1 = plt.contourf(xv, yv, p)
plt.xlabel('Cure Time (s)')
plt.ylabel('Light Intensity (W/m^2)')
plt.colorbar()
plt.title('p-value (unitless) with respect to light intensity and cure time ')
# plt.axis('square')
#plt.savefig('p_value_Variable.png')
plt.close()



h2 = plt.contourf(xv, yv, E, 10)
plt.xlabel('Cure Time (s)')
plt.ylabel('Light Intensity (W/m^2)')
numTicks = np.linspace(np.min(E),np.max(E),10)
cbar1 = plt.colorbar(ticks = numTicks)
cbar1.set_label('Elastic Modulus (MPa)')
#plt.title('Elastic Modulus (MPa) vs. Light Intensity and Cure Time')
# plt.axis('square')
#plt.savefig('elasticModulus_Variable.png')
plt.close()

h3 = plt.contourf(xv, yv, glassC, 10)
plt.xlabel('Cure Time (s)')
plt.ylabel('Light Intensity (W/m^2)')
numTicks = np.linspace(np.min(glassC),np.max(glassC),10)
cbar2 = plt.colorbar(ticks = numTicks)
cbar2.set_label('glass transition temperature (C)')
#plt.title('Elastic Modulus (MPa) vs. Light Intensity and Cure Time')
# plt.axis('square')
#plt.savefig('glass_transition.png')
plt.close()

