import jax.numpy as np
from jax.scipy import linalg
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel

PROPS_K_eq  = 0
PROPS_G_eq  = 1
PROPS_G_neq = 2
PROPS_TAU   = 3

PROPS_Krate    = 0
PROPS_Er       = 1
PROPS_R        = 2
PROPS_g1       = 3
PROPS_g2       = 4
PROPS_eta      = 5
PROPS_C1       = 6
PROPS_C2       = 7
PROPS_pgel     = 8
PROPS_refRelax = 9
constGlass = [0.01, 18959, 8.3145, 109603, 722.20, 3.73, 61000, 511.792, 0.12, 0.1]

NUM_PRONY_TERMS = -1

VISCOUS_DISTORTION = slice(0, 9)

def create_material_model_functions(variableProps, properties):
    
    density = properties.get('density')
    props = _make_properties(variableProps, properties)
    variableProps = _make_variable_properties(variableProps, properties)

    def energy_density(dispGrad, state, variableProps, dt):
        return _energy_density(dispGrad, state, variableProps, dt, props)

    def compute_initial_state(shape=(1,)):
        state = np.identity(3).ravel()
        return state

    def compute_state_new(dispGrad, state, variableProps, dt):
        state = _compute_state_new(dispGrad, state, variableProps, dt, props)
        return state

    def compute_material_qoi(dispGrad, state, variableProps, dt):
        return _compute_dissipation(dispGrad, state, variableProps, dt, props)

    return MaterialModel(compute_energy_density = energy_density,
                         compute_initial_state = compute_initial_state,
                         compute_state_new = compute_state_new,
                         compute_material_qoi = compute_material_qoi,
                         density = density)

def _make_variable_properties(variableProperties, properties):
    
    p = 1 - np.exp(-constGlass[PROPS_Krate] * variableProperties[0])
    thetaGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-p)**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    refGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-constGlass[PROPS_pgel])**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    WLF = -(constGlass[PROPS_C1]*(thetaGlass - refGlass))/(constGlass[PROPS_C2] + (thetaGlass - refGlass))
    shiftFactor = 10**WLF
    relaxTime = constGlass[PROPS_refRelax]*shiftFactor
    
    variableProps = np.array([relaxTime])

    return variableProps

def _make_properties(variableProperties, properties):

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

def _energy_density(dispGrad, state, variableProps, dt, props):
    W_eq  = _eq_strain_energy(dispGrad, props)

    Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state)
    delta_Ev = _compute_state_increment(Ee_trial, variableProps, dt, props)
    Ee = Ee_trial - delta_Ev 
    
    W_neq = _neq_strain_energy(Ee, props)
    Psi = _incremental_dissipated_energy(Ee, variableProps, dt, props)

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

def _incremental_dissipated_energy(elasticStrain, variableProps, dt, props):
    G_neq = props[PROPS_G_neq]
    tau   = variableProps #props[PROPS_TAU]
    eta   = G_neq * tau

    Me = 2. * G_neq * elasticStrain
    M_bar = TensorMath.norm_of_deviator_squared(Me)

    return 0.5 * dt * M_bar / eta

def _compute_dissipation(dispGrad, state, variableProps, dt, props):
    Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, state)
    delta_Ev = _compute_state_increment(Ee_trial, variableProps, dt, props)
    Ee = Ee_trial - delta_Ev 
    
    return _incremental_dissipated_energy(Ee, variableProps, dt, props)

def _compute_state_new(dispGrad, stateOld, variableProps, dt, props):
    Ee_trial = _compute_elastic_logarithmic_strain(dispGrad, stateOld)
    delta_Ev = _compute_state_increment(Ee_trial, variableProps, dt, props)

    Fv_old = stateOld.reshape((3, 3))
    Fv_new = linalg.expm(delta_Ev)@Fv_old
    return Fv_new.ravel()

def _compute_state_increment(elasticStrain,variableProps, dt, props):
    tau   = variableProps #props[PROPS_TAU]
    integration_factor = 1. / (1. + dt / tau)

    Ee_dev = TensorMath.dev(elasticStrain)
    return dt * integration_factor * Ee_dev / tau # dt * D

def _compute_elastic_logarithmic_strain(dispGrad, stateOld):
    F = dispGrad + np.identity(3)
    Fv_old = stateOld.reshape((3, 3))

    Fe_trial = F @ np.linalg.inv(Fv_old)
    return TensorMath.log_sqrt_symm(Fe_trial.T @ Fe_trial)
