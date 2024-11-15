from collections import namedtuple
from jax import grad
from jax import jit
from optimism import EquationSolver
# from plato_optimism import exodus_writer as ExodusWriter
# from plato_optimism import GradUtilities
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism import VTKWriter
from optimism.FunctionSpace import DofManager
from optimism.FunctionSpace import EssentialBC
from optimism.inverse import MechanicsInverse
from optimism.inverse import AdjointFunctionSpace
from optimism.material import Neohookean_VariableProps
from optimism.material import HyperViscoelastic_VariableProps
import jax.numpy as np
import numpy as onp
from scipy.sparse import linalg
from typing import Callable, NamedTuple

PROPS_K_eq     = 0
PROPS_G_eq     = 1
PROPS_G_neq    = 2
PROPS_TAU      = 3
PROPS_Krate    = 4
PROPS_Er       = 5
PROPS_R        = 6
PROPS_g1       = 7
PROPS_g2       = 8
PROPS_eta      = 9
PROPS_C1       = 10
PROPS_C2       = 11
PROPS_pgel     = 12
PROPS_refRelax = 13

# PROPS_Krate    = 0
# PROPS_Er       = 1
# PROPS_R        = 2
# PROPS_g1       = 3
# PROPS_g2       = 4
# PROPS_eta      = 5
# PROPS_C1       = 6
# PROPS_C2       = 7
# PROPS_pgel     = 8
# PROPS_refRelax = 9
constGlass = [
    0.01, 
    18959, 
    8.3145, 
    109603, 
    722.20, 
    3.73, 
    8.86, #61000, 
    101.6, #511.792, 
    0.12, 
    0.1
]

G_eq = 0.855 # MPa
K_eq = 1000*G_eq # MPa
G_neq_1 = 4.0*G_eq
tau_1   = 0.1

constant_props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
const_props = [
        constant_props['equilibrium bulk modulus'],
        constant_props['equilibrium shear modulus'],
        constant_props['non equilibrium shear modulus'],
        constant_props['relaxation time']
    ]

const_props.extend(constGlass)

constGlass = const_props[:]

input_mesh = './ellipse_test_LD_sinusoidal_1.exo'
variableProperties = ReadExodusMesh.read_exodus_mesh_element_properties(input_mesh, ['light_dose'], blockNum=1)
# print("variableProperties = %s" % variableProperties[0])


p = [0]*5
thetaGlass = p[:]
WLF = p[:]
shiftFactor = p[:]
relaxTime = p[:]

for i in range(5): #len(variableProperties)):
    p[i] = 1 - np.exp(-constGlass[PROPS_Krate] * variableProperties[i])
    thetaGlass[i] = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-p[i])**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    refGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-constGlass[PROPS_pgel])**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    WLF[i] = -(constGlass[PROPS_C1]*(thetaGlass[i] - refGlass))/(constGlass[PROPS_C2] + (thetaGlass[i] - refGlass))
    shiftFactor[i] = 10**WLF[i]
    # print("LD = %s" % variableProperties[i])
    # print("p = %s" % p)
    # print("thetaGlass = %s" % thetaGlass[i])
    # print("refGlass = %s" % refGlass)
    # print("ref = %s" % constGlass[PROPS_refRelax])
    # print("shift = %s" % shiftFactor[i] )
    # print("WLF = %s" % WLF[i] )
    relaxTime[i] = constGlass[PROPS_refRelax]*shiftFactor[i]

print("LD = %s" % variableProperties)
print("p = %s" % p)
print("thetaGlass = %s" % thetaGlass)
print("refGlass = %s" % refGlass)
print("ref = %s" % constGlass[PROPS_refRelax])
print("shift = %s" % shiftFactor)
print("WLF = %s" % WLF)





