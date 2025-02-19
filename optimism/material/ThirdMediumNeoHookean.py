import jax
import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel
from optimism import TensorMath
from optimism import ScalarRootFind

# props
PROPS_MU     = 0
PROPS_KAPPA  = 1
PROPS_LAMBDA = 2
PROPS_REGULARIZATION_CONSTANT = 3

def create_material_model_functions(properties):

    density = properties.get('density')
    props = _make_properties(properties['bulk modulus'],
                             properties['shear modulus'],
                             properties['regularization_constant'])

    def strain_energy(dispGrad, internalVars, dt):
        del internalVars
        del dt
        # return neo_hookean_energy_density(dispGrad[0:3,0:3], props)
        # return stabilized_neo_hookean_energy_density(dispGrad, props)
        # return teran_invertible_energy_density(dispGrad, props)
        return neo_hookean_energy_density(dispGrad[0:3,0:3], props) + _regularization_energy(dispGrad[0:2, 3:], props)

    def compute_state_new(dispGrad, internalVars, dt):
        del dispGrad
        del dt
        return internalVars

    def compute_material_qoi(dispGrad, internalVars, dt):
        del internalVars
        del dt
        return _compute_volumetric_jacobian(dispGrad[0:3,0:3])

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         compute_material_qoi = compute_material_qoi,
                         density = density)

def make_initial_state():
    return np.array([])

def _make_properties(kappa, mu, c):
    lamda = kappa - (2.0/3.0) * mu
    return np.array([mu, kappa, lamda, c])

def neo_hookean_energy_density(dispGrad, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    J23 = np.power(J, -2.0/3.0)
    I1Bar = J23*np.tensordot(F,F)
    Wvol = 0.5*props[PROPS_KAPPA]*(np.log(J)**2)
    Wdev = 0.5*props[PROPS_MU]*(I1Bar - 3.0)
    return Wdev + Wvol

def stabilized_neo_hookean_energy_density(dispGrad, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    I1 = np.tensordot(F,F)
    Wiso = props[PROPS_MU]/2.0 * (I1 - 3.0)
    alpha = 1.0 + props[PROPS_MU]/props[PROPS_LAMBDA]
    Wvol = props[PROPS_LAMBDA]/2.0 * (J - alpha)**2
    return Wiso + Wvol

def teran_invertible_energy_density(dispGrad, props):
    J_min = 0.01
    J = TensorMath.det(dispGrad + np.identity(3))
    return jax.lax.cond(J >= J_min, _standard_energy, _stabilizing_energy_extension, dispGrad, props, J_min)

def _standard_energy(gradDisp, props, J_min):
    I1m3 = 2*np.trace(gradDisp) + np.tensordot(gradDisp, gradDisp)
    Jm1 = TensorMath.detpIm1(gradDisp)
    logJ = np.log1p(Jm1)
    return 0.5*props[PROPS_MU]*I1m3 - props[PROPS_MU]*logJ + 0.5*props[PROPS_LAMBDA]*logJ**2

def _energy_from_principal_stretches(stretches, props):
    J = stretches[0]*stretches[1]*stretches[2]
    return 0.5*props[PROPS_MU]*(np.sum(stretches**2) - 3) - props[PROPS_MU]*np.log(J) + 0.5*props[PROPS_LAMBDA]*np.log(J)**2

def _stabilizing_energy_extension(dispGrad, props, J_min):
    F = dispGrad + np.eye(3)
    C = F.T@F
    stretches_squared, _ = TensorMath.eigen_sym33_unit(C)
    stretches = np.sqrt(stretches_squared)
    stretches = stretches.at[0].set(np.where(np.linalg.det(F) < 0, -stretches[0], stretches[0]))
    ee = stretches - 1
    I1 = ee[0] + ee[1] + ee[2]
    I2 = ee[0]*ee[1] + ee[1]*ee[2] + ee[2]*ee[0]
    I3 = ee[0]*ee[1]*ee[2]
    solver_settings = ScalarRootFind.get_settings(x_tol=1e-8)
    s, _ = ScalarRootFind.find_root(lambda x: I3*x**3 + I2*x**2 + I1*x + (1 - J_min), 0.5, np.array([0.0, 1.0]), solver_settings)
    q = 1 + s*ee # series expansion point
    h = np.linalg.norm(stretches - q)
    v = h*ee/np.linalg.norm(ee) # h*u in the paper
    W = lambda x: _energy_from_principal_stretches(x, props)
    psi0, psi1 = jax.jvp(W, (q,), (v,))
    hess = jax.hessian(W)(q)
    psi2 = 0.5*np.dot(v, hess.dot(v))
    return psi0 + psi1 + psi2

def _compute_volumetric_jacobian(dispGrad):
    F = dispGrad + np.eye(3)
    return np.linalg.det(F)

def _regularization_energy(dispHessian, props):
    def third_order_inner_product(A, B):
        return np.dot(A[:,0:2].flatten(), B[:,0:2].flatten()) + 2.0*np.dot(A[:,3], B[:,3])
    return 0.5 * props[PROPS_REGULARIZATION_CONSTANT] * third_order_inner_product(dispHessian, dispHessian)
