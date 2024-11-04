import jax.numpy as np
from jax.scipy import linalg
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel

import jax

PROPS_K_eq    = 0
PROPS_G_eq    = 1
PROPS_G_neq_1 = 2
PROPS_TAU_1   = 3
PROPS_G_neq_2 = 4
PROPS_TAU_2   = 5
PROPS_G_neq_3 = 6
PROPS_TAU_3   = 7

NUM_PRONY_TERMS = 3

VISCOUS_DISTORTION = slice(0, NUM_PRONY_TERMS * 9)

def create_material_model_functions(properties):
    
    density = properties.get('density')
    props = _make_properties(properties)

    def energy_density(dispGrad, state, dt):
        return _energy_density(dispGrad, state, dt, props)

    def compute_initial_state(shape=(1,)):
        state = np.array([])
        for n in range(NUM_PRONY_TERMS):
          state = np.hstack((state, np.identity(3).ravel()))
        return state

    def compute_state_new(dispGrad, state, dt):
        state = _compute_state_new(dispGrad, state, dt, props)
        return state

    def compute_material_qoi(dispGrad, state, dt):
        return _compute_dissipated_energy(dispGrad, state, dt, props)

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
    for n in range(NUM_PRONY_TERMS):
      print(f'  Shear modulus {n + 1}   = %s' % properties[f'non equilibrium shear modulus {n + 1}'])
      print(f'  Relaxation time {n + 1} = %s' % properties[f'relaxation time {n + 1}'])

    props = np.array([
        properties['equilibrium bulk modulus'],
        properties['equilibrium shear modulus'],
        properties['non equilibrium shear modulus 1'],
        properties['relaxation time 1'],
        properties['non equilibrium shear modulus 2'],
        properties['relaxation time 2'],
        properties['non equilibrium shear modulus 3'],
        properties['relaxation time 3']
    ])

    return props

def _energy_density(dispGrad, state, dt, props):
    W_eq  = _eq_strain_energy(dispGrad, props)
    W_neq = 0.0
    Psi = 0.0
    for n in range(NUM_PRONY_TERMS):
      state_temp = state.at[9 * n:(n + 1) * 9].get()
      Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state_temp)
      delta_Ev = _compute_state_increment(Ee_trial, dt, props, PROPS_G_neq_1 + 2 * n)
      Ee = Ee_trial - delta_Ev 
      W_neq = W_neq + _neq_strain_energy(Ee, props, PROPS_G_neq_1 + 2 * n)
    
      Dv = delta_Ev / dt
      Psi = Psi + _dissipation_potential(Dv, props, PROPS_G_neq_1 + 2 * n)

    return W_eq + W_neq + dt * Psi

def _eq_strain_energy(dispGrad, props):
    K, G = props[PROPS_K_eq], props[PROPS_G_eq]
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0 / 3.0)
    I1Bar = J23 * np.tensordot(F,F)
    Wvol = 0.5 * K * (0.5 * J**2 - 0.5 - np.log(J))
    Wdev = 0.5 * G * (I1Bar - 3.0)
    return Wdev + Wvol

def _neq_strain_energy(elasticStrain, props, prop_id):
    G_neq = props[prop_id]
    return G_neq * TensorMath.norm_of_deviator_squared(elasticStrain)

def _dissipation_potential(Dv, props, prop_id):
    G_neq = props[prop_id]
    tau   = props[prop_id + 1]
    eta   = G_neq * tau

    return eta * TensorMath.norm_of_deviator_squared(Dv)

def _compute_dissipated_energy(dispGrad, state, dt, props):
    Psi = 0.0
    for n in range(NUM_PRONY_TERMS):
      state_temp = state.at[9 * n:9 * (n + 1)].get()
      Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state_temp)
      delta_Ev = _compute_state_increment(Ee_trial, dt, props, PROPS_G_neq_1 + 2 * n)
      Dv = delta_Ev / dt
      Psi = Psi + dt * _dissipation_potential(Dv, props, PROPS_G_neq_1 + 2 * n)

    return Psi

def _compute_state_new(dispGrad, stateOld, dt, props):
    state_new = np.array([])
    for n in range(NUM_PRONY_TERMS):
      state_temp = stateOld.at[9 * n:9 * (n + 1)].get()
      Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state_temp)
      delta_Ev = _compute_state_increment(Ee_trial, dt, props, PROPS_G_neq_1 + 2 * n)

      Fv_old = state_temp.reshape((3, 3))
      Fv_new = linalg.expm(delta_Ev)@Fv_old
      state_new = np.hstack((state_new, Fv_new.ravel()))
    return state_new

def _compute_state_increment(elasticStrain, dt, props, prop_id):
    tau   = props[prop_id + 1]
    integration_factor = 1. / (1. + dt / tau)

    Ee_dev = TensorMath.dev(elasticStrain)
    return dt * integration_factor * Ee_dev / tau # dt * D

def _compute_elastic_logarithmic_strain(dispGrad, stateOld):
    F = dispGrad + np.identity(3)
    Fv_old = stateOld.reshape((3, 3))

    Fe_trial = F @ np.linalg.inv(Fv_old)
    return TensorMath.log_sqrt_symm(Fe_trial.T @ Fe_trial)
