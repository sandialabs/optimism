from optimism.JaxConfig import *
from optimism.phasefield.PhaseFieldMaterialModel import MaterialModel
from optimism import TensorMath 


# index enumerations

# material properties
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3
PROPS_GC    = 4
PROPS_L     = 5


# internal variables
STATE_UNDAMAGED_STRAIN_ENERGY_DENSITY = 0
NUM_STATE_VARS = 1

Properties = namedtuple('Properties', ['E', 'nu', 'mu', 'kappa', 'Gc', 'l'])


def create_material_model_functions(properties):
    """
    properties:
      'elastic modulus':       real
      'poisson ratio':         real
      'critical energy release rate': real
      'regularization length': real
      'kinematics':            string
          Value 'large deformations' or 'small deformations'
    """
    props = make_properties(properties['elastic modulus'],
                            properties['poisson ratio'],
                            properties['critical energy release rate'],
                            properties['regularization length'])

    finiteDeformations = True
    if 'kinematics' in properties:
        if properties['kinematics'] == 'large deformations':
            finiteDeformations = True
        elif properties['kinematics'] == 'small deformations':
            finiteDeformations = False
        else:
            raise ValueError('Unknown value specified for kinematics in ThresholdModel')
    if finiteDeformations:
        compute_strain = compute_logarithmic_strain
    else:
        compute_strain = compute_linear_strain

    def compute_energy_density(dispGrad, phase, phaseGrad, internalVars):
        strain = compute_strain(dispGrad)
        return energy_density(strain, phase, phaseGrad, props)

    def compute_output_energy_density(dispGrad, phase, phaseGrad, internalVars):
        return compute_energy_density(dispGrad, phase, phaseGrad, internalVars)

    def compute_strain_energy_density(dispGrad, phase, phaseGrad, internalVars):
        strain = compute_strain(dispGrad)
        return strain_energy_density(strain, phase, props)

    def compute_phase_potential_density(dispGrad, phase, phaseGrad, internalVars):
        return phase_potential_density(phase, phaseGrad, props)

    def compute_initial_state(shape=(1,)):
        return initial_state(shape)

    def compute_state_new(dispGrad, phase, phaseGrad, internalVars):
        strain = compute_strain(dispGrad)
        return state_new(strain, phase, props)

    return MaterialModel(compute_energy_density,
                         compute_output_energy_density,
                         compute_strain_energy_density,
                         compute_phase_potential_density,
                         compute_initial_state,
                         compute_state_new)


def make_properties(E, nu, Gc, l):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return Properties(E, nu, mu, kappa, Gc, l)


def initial_state(shape):
    W = np.array([0.0])
    return np.tile(W, shape)


def energy_density(strain, phase, phaseGrad, props):
    W = strain_energy_density(strain, phase, props)
    G = phase_potential_density(phase, phaseGrad, props)
    return W + G


def state_new(strain, phase, props):
    return np.array([strain_energy_density(strain, phase, props)])


def degradation(phase):
    return (1.-phase)*(1.-phase)


def elastic_deviatoric_free_energy(strain, phase, props):
    return degradation(phase) * props[PROPS_MU] * TensorMath.norm_of_deviator_squared(strain)


def elastic_volumetric_free_energy(strain, phase, props):
    g = np.where(np.trace(strain) > 0.0, degradation(phase), 1.0)
    return g*0.5*props[PROPS_KAPPA]*np.trace(strain)**2


def strain_energy_density(strain, phase, props):
    Wdev = elastic_deviatoric_free_energy(strain, phase, props)
    Wvol = elastic_volumetric_free_energy(strain, phase, props)
    return Wdev + Wvol


def phase_potential_density(phase, gradPhase, props):
    gradPhaseNormSquared = np.dot(gradPhase, gradPhase)
    return 3.0*props[PROPS_GC]/8.0 * (phase/props[PROPS_L] + props[PROPS_L]*gradPhaseNormSquared)


def compute_linear_strain(dispGrad):
    return TensorMath.sym(dispGrad)


def compute_logarithmic_strain(dispGrad):
    F = dispGrad + np.identity(3)
    J = np.linalg.det(F)
    traceE = np.log(J)
    CIso = J**(-2.0/3.0)*F.T@F
    devE = TensorMath.mtk_log_sqrt(CIso)
    return devE + traceE/3.0*np.identity(3)
