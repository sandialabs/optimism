import jax.numpy as np
from jax.scipy import linalg
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel

PROPS_K_eq  = 0
PROPS_G_eq  = 1
PROPS_G_neq = 2
PROPS_TAU   = 3

NUM_PRONY_TERMS = -1

VISCOUS_DISTORTION = slice(0, 9)

def create_material_model_functions(properties):
    
    density = properties.get('density')
    props = _make_properties(properties)

    def energy_density(dispGrad, state, dt):
        return _energy_density(dispGrad, state, dt, props)

    def compute_initial_state(shape=(1,)):
        state = np.identity(3).ravel()
        return state

    def compute_state_new(dispGrad, state, dt):
        state = _compute_state_new(dispGrad, state, dt, props)
        return state

    def compute_material_qoi(dispGrad, state, dt):
        return _compute_dissipation(dispGrad, state, dt, props)

    return MaterialModel(compute_energy_density = energy_density,
                         compute_initial_state = compute_initial_state,
                         compute_state_new = compute_state_new,
                         compute_material_qoi = compute_material_qoi,
                         density = density)

def _make_properties(properties):

    print('Equilibrium properties')
    print('  Bulk modulus    = %s' % properties['equilibrium bulk modulus'])
    print('  Shear modulus   = %s' % properties['equilibrium shear modulus'])
    print('Prony branch properties')
    print('  Shear modulus   = %s' % properties['non equilibrium shear modulus'])
    print('  Relaxation time = %s' % properties['relaxation time'])

    props = np.array([
        properties['equilibrium bulk modulus'],
        properties['equilibrium shear modulus'],
        properties['non equilibrium shear modulus'],
        properties['relaxation time']
    ])

    return props

def _energy_density(dispGrad, state, dt, props):
    W_eq  = _eq_strain_energy(dispGrad, props)

    Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state)
    delta_Ev = _compute_state_increment(Ee_trial, dt, props)
    Ee = Ee_trial - delta_Ev 
    
    W_neq = _neq_strain_energy(Ee, props)
    Psi = _incremental_dissipated_energy(Ee, dt, props)

    return W_eq + W_neq + Psi

def _eq_strain_energy(dispGrad, props):
    K, G = props[PROPS_K_eq], props[PROPS_G_eq]
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0 / 3.0)
    I1Bar = J23 * np.tensordot(F,F)
    Wvol = 0.5 * K * (0.5 * J**2 - 0.5 - np.log(J))
    Wdev = 0.5 * G * (I1Bar - 3.0)
    return Wdev + Wvol

def _neq_strain_energy(elasticStrain, props):
    G_neq = props[PROPS_G_neq]
    return G_neq * TensorMath.norm_of_deviator_squared(elasticStrain)

def _incremental_dissipated_energy(elasticStrain, dt, props):
    G_neq = props[PROPS_G_neq]
    tau   = props[PROPS_TAU]
    eta   = G_neq * tau

    Me = 2. * G_neq * elasticStrain
    M_bar = TensorMath.norm_of_deviator_squared(Me)

    return 0.5 * dt * M_bar / eta

def _compute_dissipation(dispGrad, state, dt, props):
    Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state)
    delta_Ev = _compute_state_increment(Ee_trial, dt, props)
    Ee = Ee_trial - delta_Ev 
    
    return _incremental_dissipated_energy(Ee, dt, props)

def _compute_state_new(dispGrad, stateOld, dt, props):
    Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, stateOld)
    delta_Ev = _compute_state_increment(Ee_trial, dt, props)

    Fv_old = stateOld.reshape((3, 3))
    Fv_new = linalg.expm(delta_Ev)@Fv_old
    return Fv_new.ravel()

def _compute_state_increment(elasticStrain, dt, props):
    tau   = props[PROPS_TAU]
    integration_factor = 1. / (1. + dt / tau)

    Ee_dev = TensorMath.dev(elasticStrain)
    return dt * integration_factor * Ee_dev / tau # dt * D

def _compute_elastic_logarithmic_strain(dispGrad, stateOld):
    F = dispGrad + np.identity(3)
    Fv_old = stateOld.reshape((3, 3))

    Fe_trial = F @ np.linalg.inv(Fv_old)
    return TensorMath.log_sqrt_symm(Fe_trial.T @ Fe_trial)
