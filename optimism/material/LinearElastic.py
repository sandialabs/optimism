import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel
from optimism import TensorMath

# props
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3


def create_material_properties(properties):
    props = _make_properties(properties['elastic modulus'],
                             properties['poisson ratio'])
    return np.array(props)


def create_material_model_functions(properties):
    if 'strain measure' in properties:
        strainMeasure = properties['strain measure']
    else:
        strainMeasure = 'linear'

    if strainMeasure == 'linear':
        _strain = linear_strain
    elif strainMeasure == 'green lagrange':
        _strain = green_lagrange_strain
    elif strainMeasure == 'logarithmic':
        _strain = log_strain
    else:
        raise ValueError('Unrecognized strain measure')
    
    def strain_energy(dispGrad, internalVars, props, dt):
        del internalVars
        del dt
        strain = _strain(dispGrad)
        return _linear_elastic_energy_density(strain, props)

    density = properties.get('density')

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         density = density)


def _make_properties(E, nu):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return np.array([E, nu, mu, kappa])


def _linear_elastic_energy_density(strain, props):
    traceStrain = np.trace(strain)
    dil = 1.0/3.0 * traceStrain
    strainDev = strain - dil*np.identity(3)
    kappa = props[PROPS_KAPPA]
    mu = props[PROPS_MU]
    return 0.5*kappa*traceStrain**2 + mu*np.tensordot(strainDev,strainDev)


def make_initial_state():
    return np.array([])


def compute_state_new(dispGrad, internalVars, props, dt):
    del dispGrad
    del props
    del dt
    return internalVars


def green_lagrange_strain(dispGrad):
    return 0.5*(dispGrad + dispGrad.T + dispGrad.T@dispGrad)


def linear_strain(dispGrad):
    return TensorMath.sym(dispGrad)


def log_strain(dispGrad):
    # Compute the deviatoric and spherical parts separately
    # to preserve the sign of J.
    Jm1 = TensorMath.detpIm1(dispGrad)
    traceStrain = np.log1p(Jm1)
    F = dispGrad + np.eye(3)
    C = F.T@F
    strain = TensorMath.log_sqrt_symm(C)
    return TensorMath.dev(strain) + traceStrain/3.0*np.identity(3)
