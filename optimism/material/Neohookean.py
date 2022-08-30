from optimism.JaxConfig import *
from optimism.material.MaterialModel import MaterialModel

# props
PROPS_E     = 0
PROPS_NU    = 1
PROPS_MU    = 2
PROPS_KAPPA = 3


def create_material_model_functions(properties):
    props = _make_properties(properties['elastic modulus'],
                             properties['poisson ratio'])

    energy_density = _adagio_neohookean
    #energy_density = _neohookean_3D_energy_density
    if 'version' in properties:
        if properties['version'] == 'adagio':
            energy_density = _adagio_neohookean
        elif properties['version'] == 'coupled':
            energy_density = _neohookean_3D_energy_density
            
    def strain_energy(dispGrad, internalVars):
        return energy_density(dispGrad, internalVars, props)

    def compute_state_new(dispGrad, internalVars):
        return _compute_state_new(dispGrad, internalVars, props)
    
    density = properties.get('density')
    
    return MaterialModel(strain_energy,
                         strain_energy,
                         make_initial_state,
                         compute_state_new,
                         density)


def _make_properties(E, nu):
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return np.array([E, nu, mu, kappa])


def _neohookean_3D_energy_density(dispGrad, internalVariables, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    
    C = F.T@F
    I1 = np.trace(C)

    C1 = 0.5*props[PROPS_MU]
    D1 = 0.125*props[PROPS_KAPPA]

    return C1*(I1-3.-2.*np.log(J)) + D1*(J - 1.0/J)**2


def _adagio_neohookean(dispGrad, internalVariables, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    #print('J = ', J)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    Wvol = 0.5*props[PROPS_KAPPA]*(0.5*J**2 - np.log(J))
    Wdev = 0.5*props[PROPS_MU]*(I1Bar - 3.0)
    return Wdev + Wvol


def make_initial_state(shape=(1,)):
    return np.zeros(shape)


def _compute_state_new(dispGrad, internalVars, props):
    return internalVars
