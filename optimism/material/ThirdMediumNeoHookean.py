import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel

# props
PROPS_MU     = 0
PROPS_KAPPA  = 1
PROPS_LAMBDA = 2

def create_material_model_functions(properties):

    density = properties.get('density')
    props = _make_properties(properties['bulk modulus'],
                             properties['shear modulus'])

    def strain_energy(dispGrad, internalVars, dt):
        del internalVars
        del dt
        # return neo_hookean_energy_density(dispGrad, props)
        return stabilized_neo_hookean_energy_density(dispGrad, props)

    def compute_state_new(dispGrad, internalVars, dt):
        del dispGrad
        del dt
        return internalVars

    def compute_material_qoi(dispGrad, internalVars, dt):
        del internalVars
        del dt
        return _compute_volumetric_jacobian(dispGrad)

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         compute_material_qoi = compute_material_qoi,
                         density = density)

def make_initial_state():
    return np.array([])

def _make_properties(kappa, mu):
    lamda = kappa - (2.0/3.0) * mu
    return np.array([mu, kappa, lamda])

def neo_hookean_energy_density(dispGrad, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    Wvol = 0.5*props[PROPS_KAPPA]*(np.log(J)**2)
    Wdev = 0.5*props[PROPS_MU]*(I1Bar - 3.0)
    return Wdev + Wvol

def stabilized_neo_hookean_energy_density(dispGrad, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    I1 = np.tensordot(F,F)
    Wiso = props[PROPS_MU]/2.0 * (I1 - 3.0)
    alpha = 1.0 + props[PROPS_MU]/props[PROPS_LAMBDA]
    Wvol = props[PROPS_LAMBDA]/2.0 * (J - alpha)**2
    return Wiso + Wvol

def _compute_volumetric_jacobian(dispGrad):
    F = dispGrad + np.eye(3)
    return np.linalg.det(F)