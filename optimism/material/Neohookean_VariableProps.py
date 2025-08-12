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

def create_material_model_functions(const_props, version = 'adagio'):
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
    designVal = props[0] # design variable value between 0 and 1

    # Linearly map design variable to light intensity percentage (same as grayscale percentage)
    # Range is 20%-100% 
    light_Intensity = 80*designVal + 20

    # Fit of Young's modulus as a function of light intensity percentage
    # Young's modulus values were obtained from experimental uniaxial tensile tests
    # For these samples we use 5 second exposure with maximum intensity of 18.5 mW/cm^2
    # and 12 hour cure at 120 C
    # Young's modulus values were fit to a hyperbolic tangent using MatCal
    # Fit has a minimization fitness of 0.000193
    E = 196.0684 * ((1 - np.exp(-0.1056 * (light_Intensity - 54.5464)))/(1 + np.exp(-0.1056 * (light_Intensity - 54.5464)))) + 228.178
   
    # Specify Poisson's ratio as a "constant prop"
    nu = const_props.get('poisson ratio')
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
