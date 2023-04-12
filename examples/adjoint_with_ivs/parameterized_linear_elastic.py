import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel
from optimism import TensorMath


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
    
    def strain_energy(dispGrad, internalVars, dt, E, nu):
        del internalVars
        del dt
        strain = _strain(dispGrad)
        return _linear_elastic_energy_density(strain, E, nu)

    density = properties.get('density')

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         density = density)


def _linear_elastic_energy_density(strain, E, nu):
    traceStrain = np.trace(strain)
    dil = 1.0/3.0 * traceStrain
    strainDev = strain - dil*np.identity(3)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    mu = 0.5*E/(1.0 + nu)
    return 0.5*kappa*traceStrain**2 + mu*np.tensordot(strainDev,strainDev)


def make_initial_state():
    return np.array([])


def compute_state_new(dispGrad, internalVars, dt, E, nu):
    del dispGrad
    del dt
    del E
    del nu
    return internalVars


def green_lagrange_strain(dispGrad):
    return 0.5*(dispGrad + dispGrad.T + dispGrad.T@dispGrad)


def linear_strain(dispGrad):
    return TensorMath.sym(dispGrad)


def log_strain(dispGrad):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    traceStrain = np.log(J)
    CIso = J**(-2.0/3.0)*F.T@F
    devStrain = TensorMath.mtk_log_sqrt(CIso)
    return devStrain + traceStrain/3.0*np.identity(3)
