import jax.numpy as np
from jax import vmap
from jax.scipy import linalg
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel


# props
PROPS_K_eq  = 0
PROPS_G_eq  = 1
PROPS_G_neq = 2
PROPS_TAU   = 3

NUM_PRONY_TERMS = -1

# isvs
VISCOUS_DISTORTION = slice(0, 9)

def create_material_model_functions(properties):
    
    # prop processing
    density = properties.get('density')
    props = _make_properties(properties)

    # energy function wrapper
    def energy_density(dispGrad, state, dt):
        return _energy_density(dispGrad, state, dt, props)

    # wrapper for state var ics
    def compute_initial_state(shape=(1,)):
        num_prony_terms = properties['number of prony terms']

        def vmap_body(_):
            return np.identity(3).ravel()

        state = np.hstack(vmap(vmap_body, in_axes=(0,))(np.arange(num_prony_terms)))
        return state

    # update state vars wrapper
    def compute_state_new(dispGrad, state, dt):
        state = _compute_state_new(dispGrad, state, dt, props)
        return state

    return MaterialModel(
        energy_density, 
        compute_initial_state,
        compute_state_new,
        density
    )

# implementation
def _make_properties(properties):
    assert properties['number of prony terms'] > 0, 'Need at least 1 prony term'
    assert 'equilibrium bulk modulus' in properties.keys()
    assert 'equilibrium shear modulus' in properties.keys()
    for n in range(1, properties['number of prony terms'] + 1):
        assert 'non equilibrium shear modulus %s' % n in properties.keys()
        assert 'relaxation time %s' % n in properties.keys()

    print('Equilibrium properties')
    print('  Bulk modulus    = %s' % properties['equilibrium bulk modulus'])
    print('  Shear modulus   = %s' % properties['equilibrium shear modulus'])

    # this is dirty, fuck jax (can't use an int from a jax numpy array or else jit tries to trace that)
    global NUM_PRONY_TERMS
    NUM_PRONY_TERMS = properties['number of prony terms']

    # first pack equilibrium properties
    props = np.array([
        properties['equilibrium bulk modulus'],
        properties['equilibrium shear modulus'],
    ])

    # props = np.hstack((props, properties['number of prony terms']))

    for n in range(1, properties['number of prony terms'] + 1):
        print('Prony branch %s properties' % n)
        print('  Shear modulus   = %s' % properties['non equilibrium shear modulus %s' % n])
        print('  Relaxation time = %s' % properties['relaxation time %s' % n])
        props = np.hstack(
            (props, np.array([properties['non equilibrium shear modulus %s' % n],
                              properties['relaxation time %s' % n]])))

    return props

def _energy_density(dispGrad, state, dt, props):
    W_eq  = _eq_strain_energy(dispGrad, props)
    W_neq = _neq_strain_energy(dispGrad, state, dt, props)
    return W_eq + W_neq

# TODO generalize to arbitrary strain energy density
def _eq_strain_energy(dispGrad, props):
    K, G = props[PROPS_K_eq], props[PROPS_G_eq]
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    Wvol = 0.5 * K * (0.5 * J**2 - 0.5 - np.log(J))
    Wdev = 0.5 *G * (I1Bar - 3.0)
    return Wdev + Wvol

def _neq_strain_energy(dispGrad, stateOld, dt, props):
    I = np.identity(3)
    F = dispGrad + I
    state_new = _compute_state_new(dispGrad, stateOld, dt, props)
    Fvs = state_new.reshape((NUM_PRONY_TERMS, 3, 3))

    def vmap_body(n, Fv):
        G_neq = props[PROPS_G_neq + 2 * n]
        # tau   = props[PROPS_TAU   + 2 * n]

        Fe = F @ np.linalg.inv(Fv)
        Ee = 0.5 * TensorMath.mtk_log_sqrt(Fe.T @ Fe)

        # viscous shearing
        # Me = 2. * G_neq * Ee
        # M_bar = TensorMath.norm_of_deviator_squared(Me)
        # visco_energy = (dt / (G_neq * tau)) * M_bar**2

        # still need another term I think
        W_neq = G_neq * TensorMath.norm_of_deviator_squared(Ee) #+ visco_energy
        return W_neq

    W_neq = np.sum(vmap(vmap_body, in_axes=(0, 0))(np.arange(NUM_PRONY_TERMS), Fvs))

    return W_neq

# state update
def _compute_state_new(dispGrad, stateOld, dt, props):
    state_inc = _compute_state_increment(dispGrad, stateOld, dt, props)
    Fv_olds   = stateOld.reshape((NUM_PRONY_TERMS, 3, 3))
    Fv_incs   = state_inc.reshape((NUM_PRONY_TERMS, 3, 3))

    def vmap_body(n, Fv_old, Fv_inc):
        Fv_new = Fv_inc @ Fv_old
        return Fv_new.ravel()

    state_new = np.hstack(vmap(vmap_body, in_axes=(0, 0, 0))(np.arange(NUM_PRONY_TERMS), Fv_olds, Fv_incs))
    return state_new

def _compute_state_increment(dispGrad, stateOld, dt, props):
    I = np.identity(3)
    F = dispGrad + I
    Fv_olds = stateOld.reshape((NUM_PRONY_TERMS, 3, 3))

    def vmap_body(n, Fv_old):
        # TODO add shift factor
        G_neq = props[PROPS_G_neq + 2 * n]
        tau   = props[PROPS_TAU + 2 * n]

        # kinematics
        Fe_trial = F @ np.linalg.inv(Fv_old)
        Ee_trial = 0.5 * TensorMath.mtk_log_sqrt(Fe_trial.T @ Fe_trial)
        Ee_dev = Ee_trial - (1. / 3.) * np.trace(Ee_trial) * I

        # updates
        integration_factor = 1. / (1. + dt / tau)

        Me = 2.0 * G_neq * Ee_dev
        Me = integration_factor * Me

        Dv = (1. / (2. * G_neq * tau)) * Me
        A  = dt * Dv

        Fv_inc = linalg.expm(A)

        return Fv_inc.ravel()

    state_inc = np.hstack(vmap(vmap_body, in_axes=(0, 0))(np.arange(NUM_PRONY_TERMS), Fv_olds))
    return state_inc