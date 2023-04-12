from functools import partial
import numpy as onp
import jax
from jax import numpy as np

import parameterized_linear_elastic as Material
#import parameterized_neohookean as Material
import nonlinear


E = 2.0
nu = 0.25
properties = {"elastic modulus": E,
              "poisson ratio": nu,
              "strain measure": "linear",
              "version": "coupled"}

material = Material.create_material_model_functions(properties)
compute_stress = jax.jit(jax.grad(material.compute_energy_density))
compute_tangents = jax.jit(jax.jacfwd(compute_stress))

strain_rate = 1e-3
max_strain = 0.5
steps = 50

t_max = max_strain/strain_rate
dt = t_max/steps

def make_strain(axial_strain, lateral_strain):
    return np.diag(np.array([axial_strain, lateral_strain, lateral_strain]))

def f(lateral_strain, axial_strain, state, E, nu):
    """Compute residual for uniaxial stress state."""
    strain = make_strain(axial_strain, lateral_strain)
    stress = compute_stress(strain, state, dt, E, nu)
    return stress[1,1]

df = jax.jacfwd(f)
solve = jax.jit(partial(nonlinear.solve, f, df))

def qoi_work(strain_new, strain_old, E, nu):
    """Compute increment of work by right-side rectangle rule."""
    stress_new = compute_stress(strain_new, internal_state, dt, E, nu)
    return np.tensordot(stress_new, strain_new - strain_old)

t = 0.0
strain = np.zeros((3,3))
strain_old = np.zeros((3,3))
internal_state = material.compute_initial_state()
work = 0.0
stress_history = onp.zeros((steps, 3, 3))
strain_history = onp.zeros((steps, 3, 3))

# Phase 1: Simulation
for i in range(1, steps):
    strain_old = strain
    strain = strain.at[0,0].add(strain_rate*dt)
    t += dt

    lat_strain = solve(strain[1,1], strain[0,0], internal_state, E, nu)

    strain = np.diag(np.array([strain[0,0], lat_strain, lat_strain]))
    stress = compute_stress(strain, internal_state, dt, E, nu)
    work += qoi_work(strain, strain_old, E, nu)
    strain_history[i] = strain
    stress_history[i] = stress

work_exact = material.compute_energy_density(strain_history[-1, :, :], internal_state, dt, E, nu)
print(f"work QOI: {work:6e}")
print(f"Exact:    {work_exact:6e}")

print(f"strain hist {strain_history[:, 0, 0]}")
print(f"stress hist {stress_history[:, 0, 0]}")

# Phase 2:
# compute sensitivity

compute_adjoint_load = jax.jit(jax.grad(qoi_work, 0))
compute_adjoint_load2 = jax.jit(jax.grad(qoi_work, 1))
compute_pseudo_load = jax.jit(jax.grad(f, (3, 4)))
compute_explicit_sensitivity = jax.jit(jax.grad(qoi_work, (2, 3)))

qoi_derivatives = np.array([0.0, 0.0])
F = 0.0 # adjoint load
for i in reversed(range(1, steps)):
    strain = strain_history[i]
    strain_old = strain_history[i-1]
    stress = stress_history[i]
    K = df(strain[1,1], strain[0,0], internal_state, E, nu)
    F -= compute_adjoint_load(strain, strain_old, E, nu)[1,1]
    W = F/K
    
    explicit_sensitivity = compute_explicit_sensitivity(strain, strain_old, E, nu)
    qoi_derivatives += np.array(explicit_sensitivity)
    pseudoloads = compute_pseudo_load(strain[1,1], strain[0,0], internal_state, E, nu)
    qoi_derivatives += W*np.array(pseudoloads)

    # part of adjoint load from strain_old in qoi
    F = -compute_adjoint_load2(strain, strain_old, E, nu)[1,1]
    
    # print("------------------------------")
    # print("iter", i)
    # print(f"strain={strain}")
    # print(f"strain_old={strain_old}")
    # print(f"stress = {stress}")
    # print(f"W = {W}, F = {F}")
    # print(f"pl = {pl_nu}")
    # print(f"dqoi={dqoi_dnu}")

dwork_dE_exact = jax.jacfwd(material.compute_energy_density, 3)(strain_history[-1, :, :], internal_state, dt, E, nu)
print(f"dqoi_dE = {qoi_derivatives[0]:6e}")
print(f"exact   = {dwork_dE_exact:6e}")

dwork_dnu_exact = jax.jacfwd(material.compute_energy_density, 4)(strain_history[-1, :, :], internal_state, dt, E, nu)
print(f"dqoi_dnu = {qoi_derivatives[1]:6e}")
print(f"exact    = {dwork_dnu_exact:6e}")
