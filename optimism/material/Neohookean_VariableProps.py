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

constProps = {'Ec': 1.059, # MPa
              'K': 0.01, # cm^2/(mW*s)
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
              'tau': 0.001, # s
              'NU': 0.48 # Poisson's ratio
              }

def create_material_model_functions(version = 'adagio'):
    
    energy_density = _adagio_neohookean
    #energy_density = _neohookean_3D_energy_density
    if version == 'adagio':
        energy_density = _adagio_neohookean
    elif version == 'coupled':
        energy_density = _neohookean_3D_energy_density

    # TODO add props as input after internalVars
    def strain_energy(dispGrad, internalVars, props, dt):
        del dt
        props = _make_properties(props)
        return energy_density(dispGrad, internalVars, props)

    # TODO add props as input
    def compute_state_new(dispGrad, internalVars, props, dt):
        del dt
        props = _make_properties(props[0], props[1])
        return _compute_state_new(dispGrad, internalVars, props)

    density = 1.0 # properties.get('density')

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         density = density)



def _make_properties(props):
    p = 1 - np.exp(-constProps['K']*props[0])
    E = (constProps['Ec'] * np.exp(constProps['b'] * (p - constProps['p_gel']))) + constProps['Ed']
    nu = props[1]
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
