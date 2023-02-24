from sys import float_info
from jax.scipy.linalg import solve, expm

from optimism.JaxConfig import *
from optimism.phasefield.PhaseFieldMaterialModel import MaterialModel
from optimism.material.J2Plastic import compute_flow_direction
from optimism.material import Hardening
from optimism import TensorMath 


# index enumerations

# material properties
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3
PROPS_GC    = 4
PROPS_PSIC  = 5
PROPS_L     = 6
PROPS_Y0    = 7
PROPS_H     = 8

# internal variables
STATE_EQPS = 0
STATE_PLASTIC_STRAIN = slice(1,1+9)
STATE_PLASTIC_DISTORTION = STATE_PLASTIC_STRAIN
NUM_STATE_VARS = 10


def create_material_model_functions(properties):
    """
    properties:
      'elastic modulus'
      'poisson ratio'
      'critical energy release rate'
      'critical strain energy density'
      'regularization length'
      'yield strength'
      'hardening modulus'
    """
    props = make_properties(properties['elastic modulus'],
                            properties['poisson ratio'],
                            properties['critical energy release rate'],
                            properties['critical strain energy density'],
                            properties['regularization length'],
                            properties['yield strength'])

    compute_elastic_strain = _compute_elastic_linear_strain
    # parse kinematics
    finiteDeformations = True
    if 'kinematics' in properties:
        if properties['kinematics'] == 'large deformations':
            finiteDeformations = True
        elif properties['kinematics'] == 'small deformations':
            finiteDeformations = False
        else:
            raise ValueError('Unknown value specified for kinematics in SandiaModel')
    if finiteDeformations:
        compute_elastic_strain = _compute_elastic_logarithmic_strain
    else:
        compute_elastic_strain = _compute_elastic_linear_strain

    hardeningModel = Hardening.create_hardening_model(properties)
    
    def compute_energy_density(dispGrad, phase, phaseGrad, internalVars, dt):
        elasticTrialStrain = compute_elastic_strain(dispGrad, internalVars)
        return energy_density_generic(elasticTrialStrain, phase, phaseGrad, internalVars, dt, props, hardeningModel, doUpdate=True)

    def compute_output_energy_density(dispGrad, phase, phaseGrad, internalVars, dt):
        elasticStrain = compute_elastic_strain(dispGrad, internalVars)
        return energy_density_generic(elasticStrain, phase, phaseGrad, internalVars, dt, props, hardeningModel, doUpdate=False)

    def compute_strain_energy_density(dispGrad, phase, phaseGrad, internalVars, dt):
        elasticStrain = compute_elastic_strain(dispGrad, internalVars)
        return strain_energy_density(elasticStrain, phase, dt, props)

    def compute_phase_potential_density(dispGrad, phase, phaseGrad, internalVars, dt):
        return phase_potential_density(phase, phaseGrad, dt, props)
    
    if finiteDeformations:
        compute_initial_state_func = make_initial_state_finite_deformations
        compute_state_new_func = compute_state_new_finite_deformations
    else:
        compute_initial_state_func = make_initial_state_small_deformations
        compute_state_new_func = compute_state_new_small_deformations

    def compute_initial_state(shape=(1,)):
        return compute_initial_state_func(shape)
        
    def compute_state_new(dispGrad, phase, phaseGrad, internalVars, dt):
        return compute_state_new_func(dispGrad, phase, phaseGrad, internalVars, dt, props, hardeningModel)
    
    return MaterialModel(compute_energy_density,
                         compute_output_energy_density,
                         compute_strain_energy_density,
                         compute_phase_potential_density,
                         compute_initial_state,
                         compute_state_new)


def make_properties(E, nu, Gc, psiC, l, Y0):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return E, nu, mu, kappa, Gc, psiC, l, Y0


def make_initial_state_finite_deformations(shape=(1,)):
    eqps = 0.0
    Fp = np.identity(3)
    pointState = np.hstack((eqps, Fp.ravel()))
    return np.tile(pointState, shape)


def make_initial_state_small_deformations(shape=(1,)):
    eqps = 0.0
    plasticStrain = np.zeros((3,3))
    pointState = np.hstack((eqps, plasticStrain.ravel()))
    return np.tile(pointState, shape)


def compute_state_new_small_deformations(dispGrad, phase, phaseGrad, stateOld, dt, props, hardeningModel):
    strain = TensorMath.sym(dispGrad)
    plasticStrainOld = stateOld[STATE_PLASTIC_STRAIN].reshape(3,3)
    elasticTrialStrain = strain - plasticStrainOld
    stateInc = compute_state_increment(elasticTrialStrain, phase, stateOld, dt, props, hardeningModel)
    eqpsNew = stateOld[STATE_EQPS] + stateInc[STATE_EQPS]
    plasticStrainNew = stateOld[STATE_PLASTIC_STRAIN] + stateInc[STATE_PLASTIC_STRAIN]
    elasticStrainNew = strain - plasticStrainNew
    return np.hstack((eqpsNew, plasticStrainNew))


def compute_state_new_finite_deformations(dispGrad, phase, phaseGrad, stateOld, dt, props, hardeningModel):
    FpOld = stateOld[STATE_PLASTIC_DISTORTION].reshape((3,3))
    elasticTrialStrain = compute_elastic_logarithmic_strain(dispGrad, FpOld)
    stateInc = compute_state_increment(elasticTrialStrain, phase, stateOld, dt, props, hardeningModel)
    eqpsNew = stateOld[STATE_EQPS] + stateInc[STATE_EQPS]
    FpOld = np.reshape(stateOld[STATE_PLASTIC_STRAIN], (3,3))
    FpNew = expm(stateInc[STATE_PLASTIC_STRAIN].reshape((3,3)))@FpOld
    elasticStrainNew = elasticTrialStrain - stateInc[STATE_PLASTIC_STRAIN].reshape((3,3))
    return np.hstack((eqpsNew, FpNew.ravel()))


def energy_density_generic(elStrain, phase, phaseGrad, state, dt, props, hardeningModel, doUpdate):
    if doUpdate:
        eqps = state[STATE_EQPS]
        N = compute_flow_direction(elStrain)
        trialStress = 2 * degradation(phase,props) * props[PROPS_MU] * np.tensordot(TensorMath.dev(elStrain), N)
        flowStress = hardeningModel.compute_flow_stress(eqps, eqps, dt)
        isYielding = trialStress > flowStress

        stateInc = np.where(isYielding,
                            update_state(elStrain, state, state, phase, dt, props, hardeningModel),
                            np.zeros(NUM_STATE_VARS))

        # Other way to introduce conditional
        # stateInc = lax.cond(isYielding,
        #                     lambda e: update_state(e, state, state, phase, props, hardeningModel),
        #                     lambda e: np.zeros(NUM_STATE_VARS),
        #                     elStrain)


    else:
        stateInc = np.zeros(NUM_STATE_VARS)
    
    eqpsNew = state[STATE_EQPS] + stateInc[STATE_EQPS]
    elasticStrainNew = elStrain - stateInc[STATE_PLASTIC_STRAIN].reshape((3,3))
        
    psi = compute_free_energy_density(elasticStrainNew, phase, phaseGrad, eqpsNew, state[STATE_EQPS], dt, props, hardeningModel)
    
    return psi


def compute_state_increment(elasticTrialStrain, phase, stateOld, dt, props, hardeningModel):
    eqpsOld = stateOld[STATE_EQPS]
    N = compute_flow_direction(elasticTrialStrain)
    trialStress = 2 * degradation(phase,props)*props[PROPS_MU] * np.tensordot(TensorMath.dev(elasticTrialStrain), N)
    flowStress = hardeningModel.compute_flow_stress(eqpsOld, eqpsOld, dt)
    isYielding = trialStress > flowStress

    stateInc = np.where(isYielding,
                        update_state(elasticTrialStrain, stateOld, stateOld, phase, props, hardeningModel),
                        np.zeros(NUM_STATE_VARS))
    
    # stateInc = lax.cond(trialStress > flowStress, 
    #                     lambda e: update_state(e, stateOld, stateOld, phase, props, hardeningModel),
    #                     lambda e: np.zeros(NUM_STATE_VARS),
    #                     elasticTrialStrain)
    
    return stateInc


def degradation(phase, props):
    gamma = 3.0 * props[PROPS_GC] / (16. * props[PROPS_L] * props[PROPS_PSIC]) - 1.
    return (1.-phase)*(1.-phase) / (1. + gamma * phase)**2

def elastic_deviatoric_free_energy(elasticStrain, phase, props):
    return degradation(phase,props)* props[PROPS_MU] * TensorMath.norm_of_deviator_squared(elasticStrain)


def elastic_volumetric_free_energy(strain, phase, props):
    g = np.where(np.trace(strain) > 0.0, degradation(phase,props), 1.0)
    return g*0.5*props[PROPS_KAPPA]*np.trace(strain)**2


def strain_energy_density(elasticStrain, phase, props):
    Wdev = elastic_deviatoric_free_energy(elasticStrain, phase, props)
    Wvol = elastic_volumetric_free_energy(elasticStrain, phase, props)
    return Wdev + Wvol


def phase_potential_density(phase, gradPhase, props):
    gradPhaseNormSquared = np.dot(gradPhase, gradPhase)
    return 3.0*props[PROPS_GC]/8.0*(phase/props[PROPS_L] + props[PROPS_L]*gradPhaseNormSquared)


def compute_free_energy_density(elasticStrain, phase, phaseGrad, eqps, eqpsOld, dt, props, hardeningModel):
    return (strain_energy_density(elasticStrain, phase, props)
            + hardeningModel.compute_hardening_energy_density(eqps, eqpsOld, dt)
            + phase_potential_density(phase, phaseGrad, props))


def incremental_potential(elasticTrialStrain, eqps, eqpsOld, phase, dt, props, hardeningModel):
    N = compute_flow_direction(elasticTrialStrain)
    elasticStrain = elasticTrialStrain - (eqps-eqpsOld) * N
    return elastic_deviatoric_free_energy(elasticStrain, phase, props) + \
        hardeningModel.compute_hardening_energy_density(eqps, eqpsOld, dt)


r = jacfwd(incremental_potential, 1)
r_and_deqps = value_and_grad(r, 1)
dr = jacfwd(r, (0,1,3))


def update_state(elasticTrialStrain, stateOld, stateNewGuess, phase, dt, props, hardeningModel):
    tol = 1e-8*props[PROPS_Y0]
    @custom_jvp
    def radial_return(eqpsGuess, etStrain, eqpsOld, phase):
        
        maxIterations=25
        def cond_func(eqpsAndIter):
            eqps,i = eqpsAndIter
            return (np.abs(r(etStrain, eqps, eqpsOld, phase, dt, props, hardeningModel)) > tol) & (i < maxIterations)
        
        def update(eqpsAndIter):
            eqps,i = eqpsAndIter
            R, dR_dEqps = r_and_deqps(etStrain, eqps, eqpsOld, phase, dt, props, hardeningModel)
            return (eqps - R / dR_dEqps, i+1)

        eqpsNew, iters = lax.while_loop(cond_func,
                                        update,
                                        (eqpsGuess,0))
        #assert(iters < maxIterations)
        return eqpsNew
    
    
    @radial_return.defjvp
    def radial_return_jvp(primals, vt):
        eqpsGuess, estrain, eqpsOld, phase = primals
        vEqpsGuess, vStrain, vEqpsOld, vPhase = vt
        eqpsNew = radial_return(eqpsGuess, estrain, eqpsOld, phase)
        drdStrain, J, drdPhase = dr(estrain, eqpsNew, eqpsOld, phase, dt, props, hardeningModel)
        tangent_out = -(drdPhase*vPhase + np.tensordot(vStrain, drdStrain) + 0.0 * vEqpsGuess) / J
        return eqpsNew, tangent_out

    eqpsOld = stateOld[STATE_EQPS]
    eqps = radial_return(stateNewGuess[STATE_EQPS], elasticTrialStrain, eqpsOld, phase)
    DeltaEqps = eqps - eqpsOld
    N = compute_flow_direction(elasticTrialStrain)
    DeltaPlasticStrain = DeltaEqps*N
    return np.hstack((DeltaEqps, DeltaPlasticStrain.ravel()))


def compute_elastic_linear_strain(dispGrad, plasticStrain):
    strain = TensorMath.sym(dispGrad)
    return strain - plasticStrain


def compute_elastic_logarithmic_strain(dispGrad, Fp):
    F = dispGrad + np.eye(3)
    FeT = solve(Fp.T, F.T)

    # Compute the deviatoric and spherical parts separately
    # to preserve the sign of J. Want to let solver sense and
    # deal with inverted elements.
    
    Je = np.linalg.det(FeT) # = J since this model is isochoric plasticity
    traceEe = np.log(Je)
    CeIso = Je**(-2./3.)*FeT@FeT.T
    EeDev = TensorMath.mtk_log_sqrt(CeIso) 
    return EeDev + traceEe/3.0*np.identity(3)


def _compute_elastic_logarithmic_strain(dispGrad, state):
    Fp = state[STATE_PLASTIC_DISTORTION].reshape((3,3))
    return compute_elastic_logarithmic_strain(dispGrad, Fp)


def _compute_elastic_linear_strain(dispGrad, state):
    plasticStrain = state[STATE_PLASTIC_STRAIN].reshape((3,3))
    return compute_elastic_linear_strain(dispGrad, plasticStrain)
