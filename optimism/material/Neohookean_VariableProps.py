import jax
import jax.numpy as np
import numpy as onp
from optimism.material.MaterialModel import MaterialModel

# props
PROPS_E      = 0
PROPS_NU     = 1
PROPS_MU     = 2
PROPS_KAPPA  = 3
PROPS_LAMBDA = 4

# constant props
CONST_PROPS_EC = 0
CONST_PROPS_K = 1
CONST_PROPS_B = 2
CONST_PROPS_PGEL = 3
CONST_PROPS_ED = 4
CONST_PROPS_ER = 5
CONST_PROPS_R = 6
CONST_PROPS_G1 = 7
CONST_PROPS_G2 = 8
CONST_PROPS_XI = 9
CONST_PROPS_C1 = 10
CONST_PROPS_C2 = 11
CONST_PROPS_RMTEMP = 12
CONST_PROPS_TAU = 13
CONST_PROPS_NU = 14

# a dict is fine for now, but we'll eventually want to 
# move this into a np.array
# dicts are slower to index then arrays which
# for small things might not seem like a big slowdown
# but can add up when the the thing is called millions of times

# TODO make these inputs to the material model
# we probably want to come up with a way
# to specify required properties and
# then pick and choose which ones to make variable
# this is fine for now though since really for this
# model it and class of materials we're interested in
# it only makes sense to change the modulus
constProps = [
    1.059, # MPa
    0.01, # cm^2/(mW*s)
    5.248, # unitless
    0.12, # unitless
    3.321, 
    18959, 
    8.314, 
    109603, # unitless
    722.2, # unitless
    3.73,# unitless
    61000, # unitless
    511.792, # K
    100, # C
    0.001, # s
    0.48 # Poisson's ratio
]

# TODO need to clean up const_props handling
# probably best way to go is to read in a dict
# and convert to an array on construction
def create_material_model_functions(const_props, version = 'adagio'):
    # TODO convert const_props from dict to list/np.array here

    if version == 'adagio':
        energy_density = _adagio_neohookean
    elif version == 'coupled':
        energy_density = _neohookean_3D_energy_density
    else:
        raise ValueError('Supported versions are \'adagio\' and \'couple\'')

    def strain_energy(dispGrad, internalVars, props, dt):
        del dt
        props = _make_properties(props, const_props)
        return energy_density(dispGrad, internalVars, props)

    def compute_state_new(dispGrad, internalVars, props, dt):
        del dt
        props = _make_properties(props, const_props)
        return _compute_state_new(dispGrad, internalVars, props)

    density = const_props.get('density')

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         density = density)


def _make_properties(props, const_props):
    p = 1 - np.exp(-constProps[CONST_PROPS_K]*props[0])
    E = (constProps[CONST_PROPS_EC] * np.exp(constProps[CONST_PROPS_B] * (p - constProps[CONST_PROPS_PGEL]))) + constProps[CONST_PROPS_ED]
    # we also want Poisson's ratio to be a "constant prop"
    # otherwise that's additional "dead" properties Ryan has to deal with
    nu = constProps[CONST_PROPS_NU]
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    lamda = E*nu/(1 + nu)/(1 - 2*nu)
    return np.array([E, nu, mu, kappa, lamda])


def _neohookean_3D_energy_density(dispGrad, internalVariables, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)

    #Wvol = 0.125*props[PROPS_LAMBDA]*(J - 1.0/J)**2
    Wvol = 0.5*props[PROPS_LAMBDA]*np.log(J)**2

    # I1m3 = tr(F.T@F) - 3, rewritten in terms of dispGrad
    I1m3 = 2.0*np.trace(dispGrad) + np.tensordot(dispGrad, dispGrad)
    C1 = 0.5*props[PROPS_MU]

    return C1*(I1m3-2.*np.log(J)) + Wvol


def _adagio_neohookean(dispGrad, internalVariables, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    #print('J = ', J)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    Wvol = 0.5*props[PROPS_KAPPA]*(0.5*J**2 - 0.5 - np.log(J))
    Wdev = 0.5*props[PROPS_MU]*(I1Bar - 3.0)
    return Wdev + Wvol


def make_initial_state():
    return np.array([])


def _compute_state_new(dispGrad, internalVars, props):
    del dispGrad
    del props
    return internalVars
