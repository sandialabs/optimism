import jax
from jax import numpy as np
import matplotlib.pyplot as plt

import parameterized_j2 as Material
import material_point


E = 2.0
nu = 0.25
Y0 = 0.01
Ysat = 0.03
eps0 = 0.125
params = np.array([E, nu, Y0, Ysat, eps0])
properties = {"kinematics": "small deformations",
              "hardening model": "voce"}

material = Material.create_material_model_functions(properties)
compute_stress = jax.jit(jax.grad(material.compute_energy_density))

strain_rate = 1e-3
max_strain = 0.5
steps = 50

def compute_plastic_work(strain_new, strain_old, isv_new, isv_old, dt, params):
    """Compute increment of dissipation by right-side rectangle rule."""
    stress_new = compute_stress(strain_new, isv_old, dt, params)
    plastic_strain_new = isv_new[Material.STATE_PLASTIC_STRAIN].reshape(3,3)
    plastic_strain_old = isv_old[Material.STATE_PLASTIC_STRAIN].reshape(3,3)
    return np.tensordot(stress_new, plastic_strain_new - plastic_strain_old)

def compute_lateral_strain(strain_new, strain_old, isv_new, isv_old, dt, params):
    return strain_new[1,1] - strain_old[1,1]

compute_qoi = compute_plastic_work

(pi, history), dpi_dp = jax.value_and_grad(material_point.simulate, 5, has_aux=True)(material, max_strain, strain_rate, steps, compute_qoi, params)
plt.plot(history.strain[:, 0, 0], history.stress[:, 0, 0])
plt.show()

dpi_dp_h = np.zeros_like(params)
for i in range(params.shape[0]):
    h = 1e-5*params[i]
    params_p = params.at[i].add(h)
    pi_p, _ = material_point.simulate(material, max_strain, strain_rate, steps, compute_qoi, params_p)
    params_m = params.at[i].add(-h)
    pi_m, _ = material_point.simulate(material, max_strain, strain_rate, steps, compute_qoi, params_m)
    dpi_dp_h = dpi_dp_h.at[i].set((pi_p - pi_m)/(2.0*h))

print(f"dpi_dp   = {dpi_dp}")
print(f"dpi_dp_h = {dpi_dp_h}")
