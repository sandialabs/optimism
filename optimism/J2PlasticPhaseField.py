from jax.lax import while_loop
from jax import custom_jvp

from optimism.JaxConfig import *
from optimism import TensorMath
from jax.scipy.linalg import solve

# props
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3
PROPS_Y0    = 4
PROPS_H     = 5

# internal variables
EQPS = 0
PLASTIC_STRAIN = slice(1,1+9)
NUM_STATE_VARS = 10


def make_properties(E, nu, Y0, H):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return (E, nu, mu, kappa, Y0, H)


def make_initial_state(shape=(1,)):
    eqps = 0.0
    plasticStrain = np.zeros((3,3))
    pointState = np.hstack((eqps, plasticStrain.ravel()))
    return np.tile(pointState, shape)


def energy_density(dispGrad, state, props, doUpdate=True):
    elasticStrain = _compute_elastic_infinitesimal_strain(dispGrad, state)
    return energy_density_generic(elasticStrain, state, props, doUpdate)


def compute_state_new(dispGrad, stateOld, props):
    elasticStrain = _compute_elastic_infinitesimal_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticStrain, stateOld, props)
    return stateOld + stateInc


def _compute_elastic_infinitesimal_strain(dispGrad, state):
    strain = TensorMath.sym(dispGrad)
    plasticStrain = state[PLASTIC_STRAIN].reshape((3,3))
    return strain - plasticStrain


# below here are inward-facing functions 
# probably will not be part of the API


def energy_density_generic(elStrain, state, props, doUpdate):
    if doUpdate:
        eqps = state[EQPS]
        N = compute_flow_direction(elStrain)
        trialStress = 2 * props[PROPS_MU] * np.tensordot(TensorMath.dev(elStrain), N)

        dummyStrain = if_then_else(trialStress > props[PROPS_Y0] + props[PROPS_H] * eqps, 
                                   elStrain,
                                   1e-8*np.identity(3))
        
        stateInc = if_then_else(trialStress > props[PROPS_Y0] + props[PROPS_H] * eqps, 
                                update_state(dummyStrain, state, state, props),
                                np.zeros(NUM_STATE_VARS))

    else:
        stateInc = np.zeros(NUM_STATE_VARS)
    
    eqpsNew = state[EQPS] + stateInc[EQPS]
    elasticStrainNew = elStrain - stateInc[PLASTIC_STRAIN].reshape((3,3))
        
    W = elastic_free_energy(elasticStrainNew, props) + \
        hardening_energy_density(eqpsNew, props)
    
    return W


def compute_logarithmic_elastic_strain(dispGrad, state):
    F = dispGrad + np.eye(3)
    Fp = state[PLASTIC_STRAIN].reshape((3,3))
    FeT = solve(Fp.T, F.T)
    Ce = FeT@FeT.T
    return TensorMath.mtk_log_sqrt(Ce)


def compute_state_increment(elasticTrialStrain, stateOld, props):
    eqpsOld = stateOld[EQPS]
    N = compute_flow_direction(elasticTrialStrain)
    trialStress = 2 * props[PROPS_MU] * np.tensordot(TensorMath.dev(elasticTrialStrain), N)
    
    stateInc = if_then_else( trialStress > props[PROPS_Y0] + props[PROPS_H] * eqpsOld, 
                             update_state(elasticTrialStrain, stateOld, stateOld, props),
                             np.zeros(NUM_STATE_VARS))
    return stateInc


def elastic_deviatoric_free_energy(elasticStrain, props):
    return props[PROPS_MU] * TensorMath.norm_of_deviator_squared(elasticStrain)


def elastic_volumetric_free_energy(strain, props):
    return 0.5*props[PROPS_KAPPA]*np.trace(strain)**2


def elastic_free_energy(elasticStrain, props):
    Wvol = elastic_volumetric_free_energy(elasticStrain, props)
    Wdev = elastic_deviatoric_free_energy(elasticStrain, props)
    return Wvol + Wdev


def hardening_energy_density(eqps, props):
    return props[PROPS_Y0] * eqps + 0.5 * props[PROPS_H] * eqps**2


def compute_flow_direction(elasticStrain):
    devElasticStrain = TensorMath.dev(elasticStrain)
    devElasticStrainNorm = np.linalg.norm(devElasticStrain, ord='fro')
    return lax.cond(devElasticStrainNorm > 1e-8,
                    lambda x: np.sqrt(3./2) * devElasticStrain / devElasticStrainNorm,
                    lambda x: np.identity(3),
                    None)


def incremental_potential(elasticTrialStrain, eqps, eqpsOld, props):
    N = compute_flow_direction(elasticTrialStrain)
    elasticStrain = elasticTrialStrain - (eqps-eqpsOld) * N
    return elastic_deviatoric_free_energy(elasticStrain, props) + \
        hardening_energy_density(eqps, props)


r = jacfwd(incremental_potential, 1)
r_and_deqps = value_and_grad(r, 1)
dr_dstrain_and_deqps = jacfwd(r, (0,1))


def update_state(elasticTrialStrain, stateOld, stateNewGuess, props):
    eqpsOld = stateOld[EQPS]
    
    tol = 1e-8
    @partial(custom_jvp, nondiff_argnums=(0,))
    def radial_return(eqpsGuess, etStrain):

        maxIterations=100
        def cond_func(eqpsAndIter):
            eqps,i = eqpsAndIter
            return (np.abs(r(etStrain, eqps, eqpsOld, props)) > tol) & (i < maxIterations)
        
        def update(eqpsAndIter):
            eqps,i = eqpsAndIter
            R, dR_dEqps = r_and_deqps(etStrain, eqps, eqpsOld, props)
            return (eqps - R / dR_dEqps, i+1)

        eqpsNew, iters = while_loop(cond_func,
                                    update,
                                    (eqpsGuess,0))
        #assert(iters < maxIterations)
        return eqpsNew
    
    
    @radial_return.defjvp
    def radial_return_jvp(eqpsGuess, estraint, vt):
        estrain, = estraint
        v, = vt
        eqpsNew = radial_return(eqpsGuess, estrain)
        drdstrain, J = dr_dstrain_and_deqps(estrain, eqpsNew, eqpsOld, props)
        
        tangent_out = -np.tensordot(v, drdstrain) / J
        return eqpsNew, tangent_out

    eqps = radial_return(stateNewGuess[EQPS], elasticTrialStrain)
    DeltaEqps = eqps - eqpsOld
    N = compute_flow_direction(elasticTrialStrain)
    DeltaPlasticStrain = DeltaEqps*N
    return np.hstack( (DeltaEqps, DeltaPlasticStrain.ravel()) )

