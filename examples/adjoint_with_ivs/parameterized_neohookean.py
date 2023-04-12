import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel


def create_material_model_functions(properties):
    energy_density = _neohookean_3D_energy_density
            
    def strain_energy(dispGrad, internalVars, dt, E, nu):
        del internalVars
        del dt
        return energy_density(dispGrad, E, nu)

    def compute_state_new(dispGrad, internalVars, dt, E, nu):
        del internalVars
        del dt
        return _compute_state_new(dispGrad, E, nu)

    density = properties.get('density')

    return MaterialModel(strain_energy,
                         make_initial_state,
                         compute_state_new,
                         density)


def _neohookean_3D_energy_density(dispGrad, E, nu):
    mu = 0.5*E/(1.0 + nu)
    lamda = E*nu/(1 + nu)/(1 - 2*nu)
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)

    #Wvol = 0.125*props[PROPS_LAMBDA]*(J - 1.0/J)**2
    Wvol = 0.5*lamda*np.log(J)**2

    # I1m3 = tr(F.T@F) - 3, rewritten in terms of dispGrad
    I1m3 = 2.0*np.trace(dispGrad) + np.tensordot(dispGrad, dispGrad)
    C1 = 0.5*mu

    return C1*(I1m3-2.*np.log(J)) + Wvol


def make_initial_state():
    return np.array([])


def _compute_state_new(dispGrad, internalVars, E, nu):
    del dispGrad
    del props
    del E
    del nu
    return internalVars
