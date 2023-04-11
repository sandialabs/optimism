from functools import partial
import numpy as onp
import jax
from jax import numpy as np
#from matplotlib import pyplot as plt

from optimism.material import LinearElastic

import nonlinear


E = 1.0
nu = 0.25
properties = {"elastic modulus": E,
              "poisson ratio": nu,
              "strain measure": "linear",
              "version": "coupled"}

material = LinearElastic.create_material_model_functions(properties)
compute_stress = jax.jit(jax.grad(material.compute_energy_density))
compute_tangents = jax.jit(jax.jacfwd(compute_stress))

strain_rate = 1e-3
max_strain = 0.5
t_max = max_strain/strain_rate

steps = 50
dt = t_max/steps

t = 0.0
strain = np.zeros((3,3))
strain_old = np.zeros((3,3))
internal_state = material.compute_initial_state()
work = 0.0

stress_history = onp.zeros((steps, 3, 3))
strain_history = onp.zeros((steps, 3, 3))

def make_strain(axial_strain, lateral_strain):
    return np.diag(np.array([axial_strain, lateral_strain, lateral_strain]))

def f(lateral_strain, axial_strain, state):
    strain = make_strain(axial_strain, lateral_strain)
    stress = compute_stress(strain, state, dt=0.0)
    return stress[1,1]

df = jax.jacfwd(f)

solve = jax.jit(partial(nonlinear.solve, f, df))

def qoi_work(stress_new, strain_new, strain_old):
    return np.tensordot(stress_new, strain_new - strain_old)

for i in range(steps):
    strain_old = strain
    strain = strain.at[0,0].add(strain_rate*dt)
    t += dt

    lat_strain = solve(strain[1,1], strain[0,0], internal_state)

    strain = np.diag(np.array([strain[0,0], lat_strain, lat_strain]))
    stress = compute_stress(strain, internal_state, dt)
    work += qoi_work(stress, strain, strain_old)
    strain_history[i] = strain
    stress_history[i] = stress

print(f"work QOI: {work:3e}")
print(f"Exact:    {0.5*E*strain[0,0]**2:3e}")

print(f"strain hist {strain_history[:, 0, 0]}")
print(f"stress hist {stress_history[:, 0, 0]}")
