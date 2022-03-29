from sys import float_info
from jax.lax import while_loop
from jax import custom_jvp

from optimism.JaxConfig import *
from optimism.material.J2Plastic import compute_flow_direction
from optimism import TensorMath 


# index enumerations

# material properties
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3
PROPS_GC    = 4
PROPS_L     = 5
PROPS_Y0    = 6
PROPS_H     = 7


# internal variables
STATE_EQPS = 0
STATE_PLASTIC_STRAIN = slice(1,1+9)
NUM_STATE_VARS = 10


Properties = namedtuple('Properties', ['E', 'nu', 'mu', 'kappa', 'Gc', 'l', 'Y0', 'H'])


def make_properties(E, nu, Gc, l, Y0, H):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return Properties(E, nu, mu, kappa, Gc, l, Y0, H)


def make_initial_state(shape=(1,)):
    eqps = 0.0
    plasticStrain = np.zeros((3,3))
    pointState = np.hstack((eqps, plasticStrain.ravel()))
    return np.tile(pointState, shape)


def energy_density(dispGrad, phase, phaseGrad, state, props, doUpdate=True):
    elasticTrialStrain = compute_elastic_strain(dispGrad, state)
    return energy_density_generic(elasticTrialStrain, phase, phaseGrad, state, props, doUpdate)


def compute_state_new(dispGrad, phase, phaseGrad, stateOld, props):
    elasticTrialStrain = compute_elastic_strain(dispGrad, stateOld)
    stateInc = compute_state_increment(elasticTrialStrain, phase, stateOld, props)
    eqpsNew = stateOld[STATE_EQPS] + stateInc[STATE_EQPS]
    plasticStrainNew = stateOld[STATE_PLASTIC_STRAIN] + stateInc[STATE_PLASTIC_STRAIN]
    return np.hstack((eqpsNew, plasticStrainNew))


## internal functions

def energy_density_generic(elStrain, phase, phaseGrad, state, props, doUpdate):
    if doUpdate:
        eqps = state[STATE_EQPS]
        N = compute_flow_direction(elStrain)
        trialStress = 2 * degradation(phase) * props[PROPS_MU] * np.tensordot(TensorMath.dev(elStrain), N)
        flowStress = flow_stress(eqps, props)

        dummyStrain = if_then_else(trialStress > flowStress, 
                                   elStrain,
                                   1e-8*np.identity(3))
        
        stateInc = if_then_else(trialStress > flowStress, 
                                update_state(dummyStrain, state, state, phase, props),
                                np.zeros(NUM_STATE_VARS))

    else:
        stateInc = np.zeros(NUM_STATE_VARS)
    
    eqpsNew = state[STATE_EQPS] + stateInc[STATE_EQPS]
    elasticStrainNew = elStrain - stateInc[STATE_PLASTIC_STRAIN].reshape((3,3))
        
    W = strain_energy_density(elasticStrainNew, phase, props) + \
        hardening_energy_density(eqpsNew, props) + phase_potential_density(phase, phaseGrad, props)
    
    return W


def compute_elastic_strain(dispGrad, state):
    strain = TensorMath.sym(dispGrad)
    plasticStrain = state[STATE_PLASTIC_STRAIN].reshape(3,3)
    return strain - plasticStrain


def compute_state_increment(elasticTrialStrain, phase, stateOld, props):
    eqpsOld = stateOld[STATE_EQPS]
    N = compute_flow_direction(elasticTrialStrain)
    trialStress = 2 * degradation(phase)*props[PROPS_MU] * np.tensordot(TensorMath.dev(elasticTrialStrain), N)
    flowStress = flow_stress(eqpsOld, props)
    
    stateInc = if_then_else( trialStress > flowStress, 
                             update_state(elasticTrialStrain, stateOld, stateOld, phase, props),
                             np.zeros(NUM_STATE_VARS))
    return stateInc


def degradation(phase):
    return (1.-phase)*(1.-phase)


def elastic_deviatoric_free_energy(elasticStrain, phase, props):
    return degradation(phase)* props[PROPS_MU] * TensorMath.norm_of_deviator_squared(elasticStrain)


def elastic_volumetric_free_energy(strain, phase, props):
    g = if_then_else(np.trace(strain) > 0.0, degradation(phase), 1.0)
    return g*0.5*props[PROPS_KAPPA]*np.trace(strain)**2


def strain_energy_density(elasticStrain, phase, props):
    Wdev = elastic_deviatoric_free_energy(elasticStrain, phase, props)
    Wvol = elastic_volumetric_free_energy(elasticStrain, phase, props)
    return Wdev + Wvol


def hardening_energy_density(eqps, props):
    return props[PROPS_Y0]*eqps + 0.5*props[PROPS_H]*eqps**2


flow_stress = jacfwd(hardening_energy_density, 0)


def phase_potential_density(phase, gradPhase, props):
    gradPhaseNormSquared = np.dot(gradPhase, gradPhase)
    return 3.0*props[PROPS_GC]/8.0*(phase/props[PROPS_L] + props[PROPS_L]*gradPhaseNormSquared)


def incremental_potential(elasticTrialStrain, eqps, eqpsOld, phase, props):
    N = compute_flow_direction(elasticTrialStrain)
    elasticStrain = elasticTrialStrain - (eqps-eqpsOld) * N
    return elastic_deviatoric_free_energy(elasticStrain, phase, props) + \
        hardening_energy_density(eqps, props)


r = jacfwd(incremental_potential, 1)
r_and_deqps = value_and_grad(r, 1)
dr_dstrain_and_deqps = jacfwd(r, (0,1))
dr = jacfwd(r, (0,1,3))


def update_state(elasticTrialStrain, stateOld, stateNewGuess, phase, props):
    eqpsOld = stateOld[STATE_EQPS]
    
    tol = 1e-8
    @partial(custom_jvp, nondiff_argnums=(0,))
    def radial_return(eqpsGuess, etStrain, phase):

        maxIterations=25
        def cond_func(eqpsAndIter):
            eqps,i = eqpsAndIter
            return (np.abs(r(etStrain, eqps, eqpsOld, phase, props)) > tol) & (i < maxIterations)
        
        def update(eqpsAndIter):
            eqps,i = eqpsAndIter
            R, dR_dEqps = r_and_deqps(etStrain, eqps, eqpsOld, phase, props)
            return (eqps - R / dR_dEqps, i+1)

        eqpsNew, iters = while_loop(cond_func,
                                    update,
                                    (eqpsGuess,0))
        #assert(iters < maxIterations)
        return eqpsNew
    
    
    @radial_return.defjvp
    def radial_return_jvp(eqpsGuess, primals, vt):
        estrain, phase = primals
        vEqps, vPhase = vt
        eqpsNew = radial_return(eqpsGuess, estrain, phase)
        drdStrain, J, drdPhase = dr(estrain, eqpsNew, eqpsOld, phase, props)
        tangent_out = -(drdPhase*vPhase + np.tensordot(vEqps, drdStrain)) / J
        return eqpsNew, tangent_out

    eqps = radial_return(stateNewGuess[STATE_EQPS], elasticTrialStrain, phase)
    DeltaEqps = eqps - eqpsOld
    N = compute_flow_direction(elasticTrialStrain)
    DeltaPlasticStrain = DeltaEqps*N
    return np.hstack( (DeltaEqps, DeltaPlasticStrain.ravel()) )


def compute_element_energy(compute_free_energy_density, U, state, shapeGrad, conn):
    nodalDisp = U[conn,:2]
    dispGrad = np.tensordot(nodalDisp, shapeGrad, axes=[0,0]) 
    dispGrad3D = np.zeros((3,3)).at[0:2,0:2].set(dispGrad)
    nodalPhase = U[conn,2]
    phaseGrad = np.tensordot(nodalPhase, shapeGrad, axes=[0,0])
    phaseGrad3D = np.zeros(3).at[0:2].set(phaseGrad)
    phase = np.average(nodalPhase)
    return compute_free_energy_density(dispGrad3D, phase, phaseGrad3D, state)


def compute_total_energy(compute_free_energy_density, U, states, mesh):
    energyDensities = vmap(compute_element_energy, (None,None,0,0,0))(compute_free_energy_density, U,
                                                                      states, mesh.shapeGrads, mesh.conns)
    return np.dot(energyDensities, mesh.vols)


def interpolate_element_kinematics(U, shapeGrad, conn):
    nodalDisp = U[conn,:2]
    dispGrad = np.tensordot(nodalDisp, shapeGrad, axes=[0,0]) 
    dispGrad = np.zeros((3,3)).at[0:2,0:2].set(dispGrad)
    
    nodalPhase = U[conn,2]
    phase = np.average(nodalPhase)
    phaseGrad = np.tensordot(nodalPhase, shapeGrad, axes=[0,0])
    phaseGrad = np.zeros(3).at[0:2].set(phaseGrad)
    return dispGrad, phase, phaseGrad
    

def interpolate_kinematics(mesh, U):
    return vmap(interpolate_element_kinematics, (None, 0, 0))(U, mesh.shapeGrads, mesh.conns)
