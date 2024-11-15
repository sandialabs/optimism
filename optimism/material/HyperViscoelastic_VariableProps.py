import jax
import jax.numpy as np
from jax.scipy import linalg
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel

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

# PROPS_Krate    = 0
# PROPS_Er       = 1
# PROPS_R        = 2
# PROPS_g1       = 3
# PROPS_g2       = 4
# PROPS_eta      = 5
# PROPS_C1       = 6
# PROPS_C2       = 7
# PROPS_pgel     = 8
# PROPS_refRelax = 9
constGlass = [
    0.01, 
    18959, 
    8.3145, 
    109603, 
    722.20, 
    3.73, 
    8.86, #61000, 
    101.6, #511.792, 
    0.12, 
    1.059,
    3.321,
    5.248,
    0.1
]

NUM_PRONY_TERMS = -1

VISCOUS_DISTORTION = slice(0, 9)

# per block
def create_material_model_functions(const_props):
    density = const_props.get('density')
    const_props = [
        const_props['equilibrium bulk modulus'],
        const_props['equilibrium shear modulus'],
        const_props['non equilibrium shear modulus'],
        const_props['relaxation time']
    ]
    const_props.extend(constGlass)

    #the props here are the props from the element, operating on quad-point
    def energy_density(dispGrad, state, props, dt):
        # jax.debug.print("props = {props}", props=props)
        props = _make_properties(props, const_props)
        # jax.debug.print("props = {props}", props=props)
        return _energy_density(dispGrad, state, dt, props)

    def compute_initial_state(shape=(1,)):
        state = np.identity(3).ravel()
        return state

    def compute_state_new(dispGrad, state, props, dt):
        props = _make_properties(props, const_props)
        state = _compute_state_new(dispGrad, state, dt, props)
        return state

    def compute_material_qoi(dispGrad, state, props, dt):
        props = _make_properties(props, const_props)
        return _compute_dissipation(dispGrad, state, dt, props)

    return MaterialModel(compute_energy_density = energy_density,
                         compute_initial_state = compute_initial_state,
                         compute_state_new = compute_state_new,
                         compute_material_qoi = compute_material_qoi,
                         density = density)

def _make_properties(variableProperties, constGlass):
    
    # p = 1 - np.exp(-constGlass[PROPS_Krate] * variableProperties[0])
    # thetaGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-p)**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    # refGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-constGlass[PROPS_pgel])**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    # WLF = -(constGlass[PROPS_C1]*(thetaGlass - refGlass))/(constGlass[PROPS_C2] + (thetaGlass - refGlass))
    # shiftFactor = 10**WLF
    # relaxTime = constGlass[PROPS_refRelax]*shiftFactor
    
    # variableProps = np.array([relaxTime])
    p = 1 - np.exp(-constGlass[PROPS_Krate] * variableProperties[0])
    thetaGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-p)**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    refGlass = constGlass[PROPS_Er]/(constGlass[PROPS_R] * np.log((constGlass[PROPS_g1]*((1-constGlass[PROPS_pgel])**constGlass[PROPS_eta])) + constGlass[PROPS_g2]))
    WLF = -(constGlass[PROPS_C1]*(thetaGlass - refGlass))/(constGlass[PROPS_C2] + (thetaGlass - refGlass))
    shiftFactor = 10**WLF
    # jax.debug.print("thetaGlass = {thetaGlass}", thetaGlass=thetaGlass)
    # jax.debug.print("refGlass = {refGlass}", refGlass=refGlass)
    # jax.debug.print("ref = {ref}", ref=constGlass[PROPS_refRelax])
    # jax.debug.print("shift = {shift}", shift=shiftFactor)
    # jax.debug.print("WLF = {WLF}", WLF=WLF)
    relaxTime = constGlass[PROPS_refRelax]*shiftFactor
    # jax.debug.print("Relax Time = {relax}", relax = relaxTime)

    # Calculate the variable shear 
    E = constGlass[PROPS_Ec]*np.exp(constGlass[PROPS_b] * (p - constGlass[PROPS_pgel])) + constGlass[PROPS_Ed]
    v = (1 - (E/(3*constGlass[PROPS_K_eq])))/2
    # jax.debug.print("Poisson's = {pois}", pois=v)
    G = E/(2*(1 + v))
    # jax.debug.print("Shear = {shear}", shear=G)
    # variableProps = np.array([relaxTime])

    # return variableProps

    # range shear from 1-1000, convert youngs to shear
    # range reference relax time from 1 to 100 

    # props = np.array([
    #     constGlass[PROPS_K_eq], 
    #     constGlass[PROPS_G_eq],
    #     constGlass[PROPS_G_neq],
    #     constGlass[PROPS_refRelax]
    # ])

    props = np.array([
        constGlass[PROPS_K_eq], 
        constGlass[PROPS_G_eq],
        constGlass[PROPS_G_neq],
        relaxTime
    ])
    
    props = np.array([
        constGlass[PROPS_K_eq], 
        0.1*G,
        4*G,
        constGlass[PROPS_refRelax]
    ])
    
    return props

# def _make_properties(variableProperties, properties):

#     print('Equilibrium properties')
#     print('  Bulk modulus    = %s' % properties['equilibrium bulk modulus'])
#     print('  Shear modulus   = %s' % properties['equilibrium shear modulus'])
#     print('Prony branch properties')
#     print('  Shear modulus   = %s' % properties['non equilibrium shear modulus'])
#     print('  Relaxation time = %s' % properties['relaxation time'])

    

#     props = np.array([
#         properties['equilibrium bulk modulus'],
#         properties['equilibrium shear modulus'],
#         properties['non equilibrium shear modulus'],
#         properties['relaxation time']
#     ])

#     return props

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
    # tau   = variableProps #props[PROPS_TAU]
    tau = props[PROPS_TAU]
    integration_factor = 1. / (1. + dt / tau)
    # jax.debug.print("tau = {tau}", tau=tau)
    # jax.debug.print("integration_factor = {integration_factor}", integration_factor=integration_factor)
    Ee_dev = TensorMath.dev(elasticStrain)
    return dt * integration_factor * Ee_dev / tau # dt * D

def _compute_elastic_logarithmic_strain(dispGrad, stateOld):
    F = dispGrad + np.identity(3)
    Fv_old = stateOld.reshape((3, 3))

    Fe_trial = F @ np.linalg.inv(Fv_old)
    return TensorMath.log_sqrt_symm(Fe_trial.T @ Fe_trial)
