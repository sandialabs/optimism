from functools import partial
import numpy as onp
import jax
from jax import numpy as np
#from matplotlib import pyplot as plt

#from optimism.material import LinearElastic
import parameterized_linear_elastic

import nonlinear


E = 1.0
nu = 0.25
properties = {"elastic modulus": E,
              "poisson ratio": nu,
              "strain measure": "linear",
              "version": "coupled"}

material = parameterized_linear_elastic.create_material_model_functions(properties)
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
    strain = make_strain(axial_strain, lateral_strain)
    stress = compute_stress(strain, state, dt, E, nu)
    return stress[1,1]

df = jax.jacfwd(f)
solve = jax.jit(partial(nonlinear.solve, f, df))

def qoi_work(strain_new, strain_old, E, nu):
    stress_new = compute_stress(strain_new, internal_state, dt, E, nu)
    return np.tensordot(stress_new, strain_new - strain_old)

t = 0.0
strain = np.zeros((3,3))
strain_old = np.zeros((3,3))
internal_state = material.compute_initial_state()
work = 0.0
stress_history = onp.zeros((steps, 3, 3))
strain_history = onp.zeros((steps, 3, 3))

for i in range(steps):
    strain_old = strain
    strain = strain.at[0,0].add(strain_rate*dt)
    t += dt

    lat_strain = solve(strain[1,1], strain[0,0], internal_state, E, nu)

    strain = np.diag(np.array([strain[0,0], lat_strain, lat_strain]))
    stress = compute_stress(strain, internal_state, dt, E, nu)
    work += qoi_work(strain, strain_old, E, nu)
    strain_history[i] = strain
    stress_history[i] = stress


print(f"work QOI: {work:3e}")
final_strain = strain[0,0]
print(f"Exact:    {0.5*E*final_strain**2:3e}")

print(f"strain hist {strain_history[:, 0, 0]}")
print(f"stress hist {stress_history[:, 0, 0]}")

compute_adjoint_load = jax.jit(jax.grad(qoi_work, 0))
compute_pseudo_load = jax.jit(jax.grad(f, 3))
compute_explicit_sensitivity = jax.jit(jax.grad(qoi_work, 2))

dqoi_dE = 0.0
for i in reversed(range(steps)):
    strain = strain_history[i]
    strain_old = strain_history[i-1]
    stress = strain_history[i]
    K = df(strain[1,1], strain[0,0], internal_state, E, nu)
    F = compute_adjoint_load(strain, strain_old, E, nu)[0,0]
    W = F/K
    dqoi_dE += compute_explicit_sensitivity(strain, strain_old, E, nu)
    psload = compute_pseudo_load(strain[1,1], strain[0,0], internal_state, E, nu)
    dqoi_dE += W*psload

print(f"dqoi_dE = {dqoi_dE}")
print(f"exact   = {0.5*final_strain**2}")
