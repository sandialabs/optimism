import numpy as onp
import jax
from jax import numpy as np
#from matplotlib import pyplot as plt

from optimism.material import LinearElastic


E = 1.0
nu = 0.0
properties = {"elastic modulus": E,
              "poisson ratio": nu,
              "strain measure": "linear",
              "version": "coupled"}

material = LinearElastic.create_material_model_functions(properties)
compute_stress = jax.jit(jax.grad(material.compute_energy_density))

strain_rate = 1e-3
max_strain = 0.5
t_max = max_strain/strain_rate

steps = 100
dt = t_max/steps

t = 0.0
ax_strain = 0.0
ax_strain_old = 0.0
internal_state = material.compute_initial_state()
work = 0.0

stress_history = onp.zeros(steps)
strain_history = onp.zeros(steps)

for i in range(steps):
    ax_strain_old = ax_strain
    ax_strain += strain_rate*dt
    t += dt

    strain = np.diag(np.array([ax_strain, 0.0, 0.0]))
    stress = compute_stress(strain, internal_state, dt)
    
    work += stress[0,0]*(ax_strain - ax_strain_old)
    strain_history[i] = ax_strain
    stress_history[i] = stress[0,0]

print(f"work QOI: {work:3e}")
print(f"Exact:    {0.5*E*ax_strain*ax_strain:3e}")

print(f"strain hist {strain_history}")
print(f"stress hist {stress_history}")
