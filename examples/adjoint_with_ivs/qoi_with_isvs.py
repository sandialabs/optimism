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

def compute_plastic_work(strain_new, strain_old, isv_new, isv_old, dt, params):
    """Compute increment of dissipation by right-side rectangle rule."""
    stress_new = compute_stress(strain_new, isv_old, dt, params)
    plastic_strain_new = isv_new[Material.STATE_PLASTIC_STRAIN].reshape(3,3)
    plastic_strain_old = isv_old[Material.STATE_PLASTIC_STRAIN].reshape(3,3)
    return np.tensordot(stress_new, plastic_strain_new - plastic_strain_old)

def compute_lateral_strain(strain_new, strain_old, isv_new, isv_old, dt, params):
    return strain_new[1,1] - strain_old[1,1]

def compute_plastic_work_exact():
    eqps = (E*max_strain - Y0)/(E + H)
    pi_exact = Y0*eqps + 0.5*H*eqps**2
    
    Y = Y0 + H*eqps
    dWp_dE = Y*(H*max_strain + Y0)/(H + E)**2
    dWp_dnu = 0.0
    dWp_dY0 = 2*eqps - max_strain #eqps - Y/(H + E)
    dWp_dH = 0.5*eqps**2 - Y*(E*max_strain - Y0)/(H + E)**2
    return pi_exact, np.array([dWp_dE, dWp_dnu, dWp_dY0, dWp_dH])
    

def compute_lateral_strain_exact():
    eqps = (E*max_strain - Y0)/(E + H) if max_strain > Y0/E else 0
    pi_exact = -nu*(max_strain - eqps) - 0.5*eqps
    
    dpi_dE = (nu - 0.5)*(H*max_strain + Y0)/(H + E)**2
    dpi_dnu = eqps - max_strain
    dpi_dY0 = (0.5 - nu)/(H + E)
    dpi_dH = (0.5 - nu)/(H + E)**2
    return pi_exact, np.array([dpi_dE, dpi_dnu, dpi_dY0, dpi_dH])
    

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--qoi",
                    choices=["plasticwork", "lateralstrain"],
                    default="plasticwork")
parser.add_argument("--steps", type=int, default=50)
args = parser.parse_args()

print(f"QoI is {args.qoi}")
if args.qoi == "plasticwork":
    compute_qoi = compute_plastic_work
    compute_qoi_exact = compute_plastic_work_exact
elif args.qoi == "lateralstrain":
    compute_qoi = compute_lateral_strain
    compute_qoi_exact = compute_lateral_strain_exact
    
steps = args.steps

(pi, history), dpi_dp = jax.value_and_grad(material_point.simulate, 5, has_aux=True)(material, max_strain, strain_rate, steps, compute_qoi, params)
pi_exact, dpi_dp_exact = compute_qoi_exact()

print("===========SUMMARY===============")
print(f"QOI:   {pi:6e}")
print(f"Exact: {pi_exact:6e}")

# plt.plot(history.strain[:, 0, 0], history.stress[:, 0, 0])
# plt.show()

print("")
print(f"dpi_dE = {dpi_dp[0]:6e}")
print(f"exact  = {dpi_dp_exact[0]:6e}")

print("")
print(f"dpi_dnu = {dpi_dp[1]:6e}")
print(f"exact   = {dpi_dp_exact[1]:6e}")

print("")
print(f"dpi_dY0  = {dpi_dp[2]:6e}")
print(f"exact    = {dpi_dp_exact[2]:6e}")

print("")
print(f"dpi_dH = {dpi_dp[3]:6e}")
print(f"exact = {dpi_dp_exact[3]:6e}")

# # Finite difference check
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
# print(f"dpi_dp_exact = {dpi_dp_exact}")
