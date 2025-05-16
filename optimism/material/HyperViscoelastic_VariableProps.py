import jax
import jax.numpy as np
from jax.scipy import linalg
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel


# define the index of key parameters
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
PROPS_Ec       = 13
PROPS_Ed       = 14
PROPS_b        = 15
PROPS_refRelax = 16

# define fixed properties
constGlass = [
    0.01,            # rate constant
    18959,           # activiation energy, J/mol
    8.3145,          # gas constants
    109603,          # g1 fitting parameter in Tg (unitless)
    722.20,          # g2 fitting parameter in Tg (unitless)
    3.73,            # \xi fitting parameter in Tg (unitless)
    8.86, #61000,    # WLF coefficient (unitless)
    101.6, #511.792, # WLF coefficient (unitless)
    0.12,            # gel point (unitless)
    1.059,           # fitting parameter for Young's modulus (unitless)
    3.321,           # fitting parameter for Young's modulus (unitless)
    5.248,           # fitting parameter for Young's modulus (unitless)
    0.1              # reference relaxation time
]

NUM_PRONY_TERMS = -1

VISCOUS_DISTORTION = slice(0, 9)

# per block
def create_material_model_functions(CONST_PROPS):
    density = CONST_PROPS.get('density')
    const_props = [
        CONST_PROPS['equilibrium bulk modulus'],
        CONST_PROPS['equilibrium shear modulus'],
        CONST_PROPS['non equilibrium shear modulus'],
        CONST_PROPS['relaxation time']
    ]
    const_props.extend(constGlass)

    #the props here are the props from the element, operating on quad-point
    def energy_density(dispGrad, state, variable_props, dt):
        props = _make_properties(variable_props, const_props)
        return _energy_density(dispGrad, state, dt, props)

    def compute_initial_state(shape=(1,)):
        state = np.identity(3).ravel()
        return state

    def compute_state_new(dispGrad, state, variable_props, dt):
        props = _make_properties(variable_props, const_props)
        state = _compute_state_new(dispGrad, state, dt, props)
        return state

    def compute_material_qoi(dispGrad, state, variable_props, dt):
        props = _make_properties(variable_props, const_props)
        return _compute_dissipation(dispGrad, state, dt, props)

    return MaterialModel(compute_energy_density = energy_density,
                         compute_initial_state = compute_initial_state,
                         compute_state_new = compute_state_new,
                         compute_material_qoi = compute_material_qoi,
                         density = density)


def _make_properties(props_variable, const_props):
    

    # extract density value
    dens = props_variable[0]
    
    # scale density to light dosage
    scaledProp = (980 * dens) + 20

    # calculate the degree of cure based on light dosage
    p = 1 - np.exp(-const_props[PROPS_Krate] * scaledProp)

    # calculate glass transition
    thetaGlass = const_props[PROPS_Er]/(const_props[PROPS_R] * np.log(const_props[PROPS_g1]*((1-p)**const_props[PROPS_eta])) + const_props[PROPS_g2])
    
    # need current temperature
    # TODO look at the props_variable and determine if we need this
    if props_variable.shape[0] > 1:
        # assert False, 'Not hooked up yet'
        temperature = props_variable[1]
    else:
        temperature = thetaGlass


    # calculate WLF 
    WLF = -(const_props[PROPS_C1]*(temperature - thetaGlass))/(const_props[PROPS_C2] + (temperature - thetaGlass))
    
    # calculate shift factor
    shiftFactor = 10**WLF
    
    # calculate relaxation time
    relaxTime = const_props[PROPS_refRelax]*shiftFactor

    # Calculate the variable shear 
    E = (const_props[PROPS_Ec]*np.exp(const_props[PROPS_b] * (p - const_props[PROPS_pgel]))) + const_props[PROPS_Ed]
    v = (3*const_props[PROPS_K_eq] - E)/(6*const_props[PROPS_K_eq])
    G = E/(2*(1 + v))
    # return np.array([const_props[PROPS_K_eq], 0.1*G, 100*G, const_props[PROPS_TAU]])
    return np.array([const_props[PROPS_K_eq], 0.1 * G, 100 * G, relaxTime])


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
    tau = props[PROPS_TAU]
    integration_factor = 1. / (1. + dt / tau)
    Ee_dev = TensorMath.dev(elasticStrain)
    return dt * integration_factor * Ee_dev / tau # dt * D

def _compute_elastic_logarithmic_strain(dispGrad, stateOld):
    F = dispGrad + np.identity(3)
    Fv_old = stateOld.reshape((3, 3))

    Fe_trial = F @ np.linalg.inv(Fv_old)
    return TensorMath.log_sqrt_symm(Fe_trial.T @ Fe_trial)
