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
    props = make_properties(properties['elastic modulus'],
                            properties['poisson ratio'],
                            properties['yield strength'])
    # parse kinematics
    finiteDeformations = True
    sethHill = False
    if 'kinematics' in properties:
        if properties['kinematics'] == 'large deformations':
            finiteDeformations = True
        elif properties['kinematics'] == 'small deformations':
            finiteDeformations = False
        elif properties['kinematics'] == 'seth hill':
            finiteDeformations = False
            sethHill = True
        else:
            raise ValueError('Unknown value specified for kinematics in J2Plastic')
        
    if finiteDeformations:
        compute_elastic_strain = compute_elastic_logarithmic_strain
    else:
        if sethHill:
            compute_elastic_strain = compute_elastic_seth_hill_strain
        else:
            compute_elastic_strain = compute_elastic_linear_strain
        
    hardeningModel = Hardening.create_hardening_model(properties)

    
    def energy_density_function(dispGrad, state):
        elasticTrialStrain = compute_elastic_strain(dispGrad, state)
        return _energy_density(elasticTrialStrain, state, props, hardeningModel)

    if finiteDeformations:
        compute_state_new_func = compute_state_new_finite_deformations
        compute_initial_state = make_initial_state_finite_deformations
    else:
        if sethHill:
            compute_state_new_func = compute_state_new_seth_hill
            compute_initial_state = make_initial_state_small_deformations
        else:
            compute_state_new_func = compute_state_new_small_deformations
            compute_initial_state = make_initial_state_small_deformations
        
    def compute_state_new_function(dispGrad, state):
            return compute_state_new_func(dispGrad, state, props, hardeningModel)

    density = properties.get('density')

    return MaterialModel(energy_density_function,
                         compute_initial_state,
                         compute_state_new_function,
                         density)


def make_properties(E, nu, Y0):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return (E, nu, mu, kappa, Y0)


def make_initial_state_finite_deformations(shape=(1,)):
    eqps = 0.0
    Fp = np.identity(3)
    return np.hstack((eqps, Fp.ravel()))


def make_initial_state_small_deformations(shape=(1,)):
    eqps = 0.0
    plasticStrain = np.zeros((3,3))
    pointState = np.hstack((eqps, plasticStrain.ravel()))
    return np.tile(pointState, shape)


def compute_state_new_finite_deformations(dispGrad, stateOld, props, hardening_model):
    elasticTrialStrain = compute_elastic_logarithmic_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticTrialStrain, stateOld, props, hardening_model)
    eqpsNew = stateOld[EQPS] + stateInc[EQPS]
    FpOld = np.reshape(stateOld[PLASTIC_DISTORTION], (3,3))
    FpNew = linalg.expm(stateInc[PLASTIC_DISTORTION].reshape((3,3)))@FpOld
    return np.hstack((eqpsNew, FpNew.ravel()))


def compute_state_new_small_deformations(dispGrad, stateOld, props, hardening_model):
    elasticTrialStrain = compute_elastic_linear_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticTrialStrain, stateOld, props, hardening_model)
    return stateOld + stateInc


def compute_state_new_seth_hill(dispGrad, stateOld, props, hardening_model):
    elasticTrialStrain = compute_elastic_seth_hill_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticTrialStrain, stateOld, props, hardening_model)
    return stateOld + stateInc


def _energy_density(elStrain, state, props, hardening_model):
    stateInc = compute_state_increment(elStrain, state, props, hardening_model)
    
    eqpsNew = state[EQPS] + stateInc[EQPS]
    elasticStrainNew = elStrain - stateInc[PLASTIC_DISTORTION].reshape((3,3))
        
    W = elastic_free_energy(elasticStrainNew, props) + hardening_model[ENERGY_DENSITY](eqpsNew)
    
    return W


def compute_state_increment(elasticStrain, state, props, hardening_model):
    eqps = state[EQPS]
    N = compute_flow_direction(elasticStrain)
    trialStress = 2 * props[PROPS_MU] * np.tensordot(TensorMath.dev(elasticStrain), N)
    flowStress = hardening_model[FLOW_STRESS](eqps)

    # The tolerance on the yield check is the same as that in the nonlinear solve.
    # This ensures that we take the elastic branch if the state vars are updated.
    isYielding = trialStress - flowStress > _TOLERANCE*props[PROPS_Y0]

    stateInc = jax.lax.cond(isYielding,
                            lambda e: update_state(e, state, state, props, hardening_model),
                            lambda e: np.zeros(NUM_STATE_VARS),
                            elasticStrain)

    return stateInc


def elastic_deviatoric_free_energy(elasticStrain, props):
    return props[PROPS_MU] * TensorMath.norm_of_deviator_squared(elasticStrain)


def elastic_volumetric_free_energy(strain, props):
    return 0.5*props[PROPS_KAPPA]*np.trace(strain)**2


def elastic_free_energy(elasticStrain, props):
    Wvol = elastic_volumetric_free_energy(elasticStrain, props)
    Wdev = elastic_deviatoric_free_energy(elasticStrain, props)
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


def incremental_potential(elasticTrialStrain, eqps, eqpsOld, props, hardening_model):
    N = compute_flow_direction(elasticTrialStrain)
    elasticStrain = elasticTrialStrain - (eqps-eqpsOld) * N
    return elastic_deviatoric_free_energy(elasticStrain, props) + \
        hardening_model[ENERGY_DENSITY](eqps)


r = jax.jacfwd(incremental_potential, 1)


def update_state(elasticTrialStrain, stateOld, stateNewGuess, props, hardening_model):
    settings = ScalarRootFind.get_settings(x_tol=0, r_tol=_TOLERANCE*props[PROPS_Y0])
    eqpsOld = stateOld[EQPS]
    eqpsGuess = stateNewGuess[EQPS]

    N = compute_flow_direction(elasticTrialStrain)
    lb = eqpsOld
    trialMises = 2 * props[PROPS_MU] * np.tensordot(TensorMath.dev(elasticTrialStrain), N)
    ub = eqpsOld + (trialMises - hardening_model.compute_flow_stress(eqpsOld))/(3.0*props[PROPS_MU])
    eqps, _ = ScalarRootFind.find_root(lambda e: r(elasticTrialStrain, e, eqpsOld, props, hardening_model),
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


# default to the Tupek strain measure
def compute_elastic_seth_hill_strain(dispGrad, state):
    m=0.25
    C = dispGrad.T@dispGrad
    strain = (TensorMath.mtk_pow(C,m) - np.identity(3)) / (2*m)
    plasticStrain = state[PLASTIC_STRAIN].reshape((3,3))
    return strain - plasticStrain
