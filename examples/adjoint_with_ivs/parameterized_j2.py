import jax
import jax.numpy as np
from jax.scipy import linalg

from optimism.material.MaterialModel import MaterialModel
from optimism.material import Hardening
from optimism import TensorMath
from optimism import ScalarRootFind


# props
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3
PROPS_Y0    = 4

# internal variables
EQPS = 0
PLASTIC_DISTORTION = slice(1,1+9)
PLASTIC_STRAIN = PLASTIC_DISTORTION # alias for better readability in small deformation theory
NUM_STATE_VARS = 10

# hardening functions
ENERGY_DENSITY  = 0
FLOW_STRESS     = 1

# tolerance on EPQS change
_TOLERANCE = 1e-10

def create_material_model_functions(properties):
    # parse kinematics
    finiteDeformations = True
    sethHill = False
    if 'kinematics' in properties:
        if properties['kinematics'] == 'large deformations':
            finiteDeformations = True
        elif properties['kinematics'] == 'small deformations':
            finiteDeformations = False
        else:
            raise ValueError('Unknown value specified for kinematics in J2Plastic')
        
    if finiteDeformations:
        compute_elastic_strain = compute_elastic_logarithmic_strain
    else:
        compute_elastic_strain = compute_elastic_linear_strain
    
    def energy_density_function(dispGrad, state, dt, E, nu, Y0, H):
        elasticTrialStrain = compute_elastic_strain(dispGrad, state)
        return _energy_density(elasticTrialStrain, state, dt, E, nu, Y0, H)

    if finiteDeformations:
        compute_state_new_func = compute_state_new_finite_deformations
        compute_initial_state = make_initial_state_finite_deformations
    else:
        compute_state_new_func = compute_state_new_small_deformations
        compute_initial_state = make_initial_state_small_deformations
        
    def compute_state_new_function(dispGrad, state, dt, E, nu, Y0, H):
        return compute_state_new_func(dispGrad, state, dt, E, nu, Y0, H)

    density = properties.get('density')

    return MaterialModel(energy_density_function,
                         compute_initial_state,
                         compute_state_new_function,
                         density)


def linear_hardening_potential(eqps, Y0, H):
    return Y0*eqps + 0.5*H*eqps**2


def power_law_hardening_potential(eqps, Y0, n, eps0):
    A = n*Y0*eps0/(1.0 + n)
    x = eqps/eps0
    return A*( (1.0 + x)**((n+1)/n) - 1.0 )


def hardening_potential(eqps_new, eqps_old, dt, Y0, H):
    return linear_hardening_potential(eqps_new, Y0, H)


def make_initial_state_finite_deformations(shape=(1,)):
    eqps = 0.0
    Fp = np.identity(3)
    return np.hstack((eqps, Fp.ravel()))


def make_initial_state_small_deformations(shape=(1,)):
    eqps = 0.0
    plasticStrain = np.zeros((3,3))
    pointState = np.hstack((eqps, plasticStrain.ravel()))
    return np.tile(pointState, shape)


def compute_state_new_finite_deformations(dispGrad, stateOld, dt, E, nu, Y0, H):
    elasticTrialStrain = compute_elastic_logarithmic_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticTrialStrain, stateOld, dt, E, nu, Y0, H)
    eqpsNew = stateOld[EQPS] + stateInc[EQPS]
    FpOld = np.reshape(stateOld[PLASTIC_DISTORTION], (3,3))
    FpNew = linalg.expm(stateInc[PLASTIC_DISTORTION].reshape((3,3)))@FpOld
    return np.hstack((eqpsNew, FpNew.ravel()))


def compute_state_new_small_deformations(dispGrad, stateOld, dt, E, nu, Y0, H):
    elasticTrialStrain = compute_elastic_linear_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticTrialStrain, stateOld, dt, E, nu, Y0, H)
    return stateOld + stateInc


def _energy_density(elStrain, state, dt, E, nu, Y0, H):
    stateInc = compute_state_increment(elStrain, state, dt, E, nu, Y0, H)
    
    eqpsNew = state[EQPS] + stateInc[EQPS]
    elasticStrainNew = elStrain - stateInc[PLASTIC_DISTORTION].reshape((3,3))
        
    W = elastic_free_energy(elasticStrainNew, E, nu) + hardening_potential(eqpsNew, state[EQPS], dt, Y0, H)
    
    return W


def compute_state_increment(elasticStrain, state, dt, E, nu, Y0, H):
    eqps = state[EQPS]
    N = compute_flow_direction(elasticStrain)
    mu = 0.5*E/(1.0 + nu)
    trialStress = 2 * mu * np.tensordot(TensorMath.dev(elasticStrain), N)
    flowStress = jax.jacfwd(hardening_potential)(eqps, eqps, dt, Y0, H)

    # The tolerance on the yield check is the same as that in the nonlinear solve.
    # This ensures that we take the elastic branch if the state vars are updated.
    isYielding = trialStress - flowStress > _TOLERANCE*Y0

    stateInc = jax.lax.cond(isYielding,
                            lambda e: update_state(e, state, dt, E, nu, Y0, H),
                            lambda e: np.zeros(NUM_STATE_VARS),
                            elasticStrain)

    return stateInc


def elastic_deviatoric_free_energy(elasticStrain, E, nu):
    mu = 0.5*E/(1.0 + nu)
    return mu * TensorMath.norm_of_deviator_squared(elasticStrain)


def elastic_volumetric_free_energy(strain, E, nu):
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return 0.5*kappa*np.trace(strain)**2


def elastic_free_energy(elasticStrain, E, nu):
    Wvol = elastic_volumetric_free_energy(elasticStrain, E, nu)
    Wdev = elastic_deviatoric_free_energy(elasticStrain, E, nu)
    return Wvol + Wdev


def compute_flow_direction(elasticStrain):
    devElasticStrain = TensorMath.dev(elasticStrain)
    devElasticStrainNormSquared = np.tensordot(devElasticStrain, devElasticStrain)
    isNonzero = devElasticStrainNormSquared > 1e-16
    devElasticStrainNormSquared = np.where(isNonzero,
                                           devElasticStrainNormSquared,
                                           1.0)
    dummyN = 0.5*np.array([[0.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 0.0]])
    return np.where(isNonzero,
                    np.sqrt(3./2.)/np.sqrt(devElasticStrainNormSquared) * devElasticStrain,
                    dummyN)


def incremental_potential(elasticTrialStrain, eqps, eqpsOld, dt, E, nu, Y0, H):
    N = compute_flow_direction(elasticTrialStrain)
    elasticStrain = elasticTrialStrain - (eqps-eqpsOld) * N
    return elastic_deviatoric_free_energy(elasticStrain, E, nu) + \
        hardening_potential(eqps, eqpsOld, dt, Y0, H)


r = jax.jacfwd(incremental_potential, 1)


def update_state(elasticTrialStrain, stateOld, dt, E, nu, Y0, H):
    settings = ScalarRootFind.get_settings(x_tol=0, r_tol=_TOLERANCE*Y0)
    eqpsOld = stateOld[EQPS]

    N = compute_flow_direction(elasticTrialStrain)
    lb = eqpsOld
    mu = 0.5*E/(1.0 + nu)
    trialMises = 2 * mu * np.tensordot(TensorMath.dev(elasticTrialStrain), N)
    ub = eqpsOld + (trialMises - jax.jacfwd(hardening_potential)(eqpsOld, eqpsOld, dt, Y0, H))/(3.0*mu)
    # Avoid the initial guess eqpsGuess = eqpsOld, because the power law rate sensitivity has an infinte slope
    # in this case.
    eqpsGuess = 0.5*(lb + ub)
    eqps, _ = ScalarRootFind.find_root(lambda e: r(elasticTrialStrain, e, eqpsOld, dt, E, nu, Y0, H),
                                       eqpsGuess,
                                       np.array([lb, ub]),
                                       settings)
    DeltaEqps = eqps - eqpsOld
    DeltaPlasticStrain = DeltaEqps*N
    return np.hstack( (DeltaEqps, DeltaPlasticStrain.ravel()) )


def compute_elastic_logarithmic_strain(dispGrad, state):
    F = dispGrad + np.eye(3)
    Fp = state[PLASTIC_DISTORTION].reshape((3,3))
    FeT = linalg.solve(Fp.T, F.T)

    # Compute the deviatoric and spherical parts separately
    # to preserve the sign of J. Want to let solver sense and
    # deal with inverted elements.
    
    Je = np.linalg.det(FeT) # = J since this model is isochoric plasticity
    traceEe = np.log(Je)
    CeIso = Je**(-2./3.)*FeT@FeT.T
    EeDev = TensorMath.mtk_log_sqrt(CeIso) 
    return EeDev + traceEe/3.0*np.identity(3)


def compute_elastic_linear_strain(dispGrad, state):
    strain = TensorMath.sym(dispGrad)
    plasticStrain = state[PLASTIC_STRAIN].reshape((3,3))
    return strain - plasticStrain

