from functools import partial
import numpy as onp
import jax
from jax import numpy as np
import matplotlib.pyplot as plt

import parameterized_j2 as Material
import nonlinear


E = 2.0
nu = 0.25
Y0 = 0.01
H = E/50.0
properties = {"elastic modulus": E,
              "poisson ratio": nu,
              "kinematics": "small deformations"}

material = Material.create_material_model_functions(properties)
compute_stress = jax.jit(jax.grad(material.compute_energy_density))
compute_tangents = jax.jit(jax.jacfwd(compute_stress))
compute_state_new = jax.jit(material.compute_state_new)
compute_energy_density = jax.jit(material.compute_energy_density)

strain_rate = 1e-3
max_strain = 0.5
steps = 50

t_max = max_strain/strain_rate
dt = t_max/steps

def make_strain(axial_strain, lateral_strain):
    return np.diag(np.array([axial_strain, lateral_strain, lateral_strain]))

def f(lateral_strain, axial_strain, state, E, nu, Y0, H):
    """Compute residual for uniaxial stress state."""
    strain = make_strain(axial_strain, lateral_strain)
    stress = compute_stress(strain, state, dt, E, nu, Y0, H)
    return stress[1,1]

df = jax.jacfwd(f)
solve = jax.jit(partial(nonlinear.solve, f, df))

def qoi(strain_new, isv_new, isv_old, E, nu, Y0, H):
    """Compute increment of dissipation by right-side rectangle rule."""
    stress_new = compute_stress(strain_new, isv_old, dt, E, nu, Y0, H)
    plastic_strain_new = isv_new[Material.PLASTIC_STRAIN].reshape(3,3)
    plastic_strain_old = isv_old[Material.PLASTIC_STRAIN].reshape(3,3)
    return np.tensordot(stress_new, plastic_strain_new - plastic_strain_old)

t = 0.0
strain = np.zeros((3,3))
strain_old = np.zeros((3,3))
internal_state = material.compute_initial_state()
D = 0.0
stress_history = onp.zeros((steps, 3, 3))
strain_history = onp.zeros((steps, 3, 3))

# Phase 1: Simulation
for i in range(1, steps):
    strain_old = strain
    strain = strain.at[0,0].add(strain_rate*dt)
    t += dt

    lat_strain = solve(strain[1,1], strain[0,0], internal_state, E, nu, Y0, H)

    strain = np.diag(np.array([strain[0,0], lat_strain, lat_strain]))
    stress = compute_stress(strain, internal_state, dt, E, nu, Y0, H)
    internal_state_old = internal_state
    internal_state = compute_state_new(strain, internal_state, dt, E, nu, Y0, H)
    D += qoi(strain, internal_state, internal_state_old, E, nu, Y0, H)
    strain_history[i] = strain
    stress_history[i] = stress

D_exact = Material.linear_hardening_potential(internal_state[Material.EQPS], Y0, H)
compute_energy_density(strain_history[-1, :, :], internal_state, dt, E, nu, Y0, H)
print(f"QOI: {D:6e}")
print(f"Exact:    {D_exact:6e}")

print(f"strain hist {strain_history[:, 0, 0]}")
print(f"stress hist {stress_history[:, 0, 0]}")

plt.plot(strain_history[:, 0, 0], stress_history[:, 0, 0])
plt.show()

# Phase 2:
# compute sensitivity

# compute_adjoint_load = jax.jit(jax.grad(qoi, 0))
# compute_adjoint_load2 = jax.jit(jax.grad(qoi, 1))
# compute_pseudo_load = jax.jit(jax.grad(f, (3, 4)))
# compute_explicit_sensitivity = jax.jit(jax.grad(qoi, (2, 3)))

# qoi_derivatives = np.array([0.0, 0.0])
# F = 0.0 # adjoint load
# for i in reversed(range(1, steps)):
#     strain = strain_history[i]
#     strain_old = strain_history[i-1]
#     stress = stress_history[i]
#     K = df(strain[1,1], strain[0,0], internal_state, E, nu)
#     F -= compute_adjoint_load(strain, strain_old, E, nu)[1,1]
#     W = F/K
    
#     explicit_sensitivity = compute_explicit_sensitivity(strain, strain_old, E, nu)
#     qoi_derivatives += np.array(explicit_sensitivity)
#     pseudoloads = compute_pseudo_load(strain[1,1], strain[0,0], internal_state, E, nu)
#     qoi_derivatives += W*np.array(pseudoloads)

#     # part of adjoint load from strain_old in qoi
#     F = -compute_adjoint_load2(strain, strain_old, E, nu)[1,1]
    
#     # print("------------------------------")
#     # print("iter", i)
#     # print(f"strain={strain}")
#     # print(f"strain_old={strain_old}")
#     # print(f"stress = {stress}")
#     # print(f"W = {W}, F = {F}")
#     # print(f"pl = {pl_nu}")
#     # print(f"dqoi={dqoi_dnu}")

# dwork_dE_exact = jax.jacfwd(material.compute_energy_density, 3)(strain_history[-1, :, :], internal_state, dt, E, nu)
# print(f"dqoi_dE = {qoi_derivatives[0]:6e}")
# print(f"exact   = {dwork_dE_exact:6e}")

# dwork_dnu_exact = jax.jacfwd(material.compute_energy_density, 4)(strain_history[-1, :, :], internal_state, dt, E, nu)
# print(f"dqoi_dnu = {qoi_derivatives[1]:6e}")
# print(f"exact    = {dwork_dnu_exact:6e}")