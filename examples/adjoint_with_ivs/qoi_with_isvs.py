"""Compute one of several example quantitities of interest for a material with state."""
import argparse

import jax
from jax import numpy as np
import matplotlib.pyplot as plt

import parameterized_j2 as Material
import material_point


E = 2.0
nu = 0.25
Y0 = 0.01
H = E/50.0
params = np.array([E, nu, Y0, H])
properties = {"kinematics": "small deformations",
              "hardening model": "linear"}

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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--qoi",
                    choices=["plasticwork", "lateralstrain"],
                    default="plasticwork")
args = parser.parse_args()

print(f"QoI is {args.qoi}")
if args.qoi == "plasticwork":
    compute_qoi = compute_plastic_work
elif args.qoi == "lateralstrain":
    compute_qoi = compute_lateral_strain

# pi, (history, _) = material_point.simulate_fwd(material, max_strain, strain_rate, steps, compute_lateral_strain, params)
# strain_history = history[0]
# stress_history = history[1]
# plt.plot(strain_history[:, 0, 0], stress_history[:, 0, 0])
# plt.show()

pi, dpi_dp = jax.value_and_grad(material_point.simulate, 5)(material, max_strain, strain_rate, steps, compute_qoi, params)

eqps = (E*max_strain - Y0)/(E + H)
pi_exact = Y0*eqps + 0.5*H*eqps**2

print("===========SUMMARY===============")
print(f"Plastic work: {pi:6e}")
print(f"Exact:        {pi_exact:6e}")

# print(f"strain hist {strain_history[:, 0, 0]}")
# print(f"stress hist {stress_history[:, 0, 0]}")
# plt.plot(strain_history[:, 0, 0], stress_history[:, 0, 0])
# plt.show()

print("")
print(f"dpi_dE = {dpi_dp[0]:6e}")
print(f"exact  = {(Y0+H*eqps)*(H*max_strain+Y0)/(H+E)**2:6e}")

print("")
print(f"dpi_dnu = {dpi_dp[1]:6e}")
print(f"exact   = 0")

dD_dY0_exact = eqps - (Y0+H*eqps)/(H+E)
print("")
print(f"dpi_dY0  = {dpi_dp[2]:6e}")
print(f"exact    = {dD_dY0_exact:6e}")

dpi_dH_exact = 0.5*eqps**2-H*eqps*(E*max_strain-Y0)/(H+E)**2
print("")
print(f"dpi_dH = {dpi_dp[3]:6e}")
print(f"exact = {dpi_dH_exact:6e}")

# dpi_dp_h = np.zeros_like(params)
# for i in range(params.shape[0]):
#     h = 1e-5*params[i]
#     params_p = params.at[i].add(h)
#     pi_p = material_point.simulate(material, max_strain, strain_rate, steps, compute_qoi, params_p)
#     params_m = params.at[i].add(-h)
#     pi_m = material_point.simulate(material, max_strain, strain_rate, steps, compute_qoi, params_m)
#     dpi_dp_h = dpi_dp_h.at[i].set((pi_p - pi_m)/(2.0*h))

# print(f"dpi_dp   = {dpi_dp}")
# print(f"dpi_dp_h = {dpi_dp_h}")
