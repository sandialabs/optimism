from functools import partial
import numpy as onp
import jax
from jax import numpy as np
import matplotlib.pyplot as plt

import parameterized_j2 as Material
import nonlinear


def simulate(material, max_strain, strain_rate, steps, compute_qoi, *params):
    t_max = max_strain/strain_rate
    dt = t_max/steps
    compute_stress = jax.jit(jax.grad(material.compute_energy_density))
    #compute_tangents = jax.jit(jax.jacfwd(compute_stress))
    compute_state_new = jax.jit(material.compute_state_new)
    #compute_energy_density = jax.jit(material.compute_energy_density)


    def f(lateral_strain, axial_strain, state, *params):
        """Compute residual for uniaxial stress state."""
        strain = np.diag(np.array([axial_strain, lateral_strain, lateral_strain]))
        stress = compute_stress(strain, state, dt, *params)
        return stress[1,1]

    df = jax.jacfwd(f)
    solve = jax.jit(partial(nonlinear.solve, f, df))

    t = 0.0
    strain = np.zeros((3,3))
    internal_state = material.compute_initial_state()
    pi = 0.0
    stress_history = onp.zeros((steps, 3, 3))
    strain_history = onp.zeros((steps, 3, 3))
    internal_state_history = onp.zeros((steps, internal_state.shape[0]))
    internal_state_history[0] = internal_state

    # Phase 1: Simulation
    for i in range(1, steps):
        print("-----------------------")
        print(f"Step {i}")
        strain_old = strain
        strain = strain.at[0,0].add(strain_rate*dt)
        t += dt

        lat_strain = solve(strain[1,1], strain[0,0], internal_state, *params)

        # post-process and save data needed for reverse pass
        strain = np.diag(np.array([strain[0,0], lat_strain, lat_strain]))
        stress = compute_stress(strain, internal_state, dt, *params)
        internal_state_old = internal_state
        internal_state = compute_state_new(strain, internal_state, dt, *params)
        pi += compute_qoi(strain, internal_state, internal_state_old, dt, *params)
        strain_history[i] = strain
        stress_history[i] = stress
        internal_state_history[i] = internal_state

        print(f"residual norm: {np.abs(stress[1,1]) })")
        print(f"strain = {strain[0,0]}, \tstress = {stress[0,0]}")
        print(f"QOI: {pi:6e}")

    return (strain_history, stress_history, internal_state_history), pi



# Phase 2:
# compute sensitivity

# compute_dpi = jax.jit(jax.grad(qoi, (0, 1, 2, 5)))
# compute_dq = jax.jit(jax.jacrev(compute_state_new, (0, 1, 5)))
# compute_dsigma = jax.jit(jax.jacrev(compute_stress, (0, 1, 5)))

# qoi_derivatives = 0.0
# adjoint_internal_state_force = np.zeros_like(internal_state)
# F = 0.0 # adjoint load
# print("\n\n")
# print("ADJOINT PHASE")
# for i in reversed(range(1, steps)):
#     print("-------------------------")
#     print(f"Step {i}")
#     strain = strain_history[i]
#     strain_old = strain_history[i-1]
#     stress = stress_history[i]
#     internal_state = internal_state_history[i]
#     internal_state_old = internal_state_history[i-1]
    
#     dpi_dH, dpi_dQ, dpi_dQOld, dpi_dY0 = compute_dpi(strain, internal_state, internal_state_old, E, nu, Y0, H)
#     dQ_dH, dQ_dQOld, dQ_Y0 = compute_dq(strain, internal_state_old, dt, E, nu, Y0, H)
#     K = df(strain[1,1], strain[0,0], internal_state_old, E, nu, Y0, H)
#     F = -dpi_dH[1,1]
#     adjoint_internal_state_force += dpi_dQ
#     F -= np.tensordot(adjoint_internal_state_force, dQ_dH, axes=1)[1,1]
#     W = F/K

#     qoi_derivatives += dpi_dY0
#     dSigma_dH, dSigma_dQOld, dSigma_dY0 = compute_dsigma(strain, internal_state_old, dt, E, nu, Y0, H)
#     qoi_derivatives += W*dSigma_dY0[1,1]
#     qoi_derivatives += np.dot(adjoint_internal_state_force, dQ_Y0)

#     adjoint_internal_state_force = np.tensordot(adjoint_internal_state_force, dQ_dQOld, axes=1)
#     adjoint_internal_state_force += W*dSigma_dQOld[1,1]
#     adjoint_internal_state_force += dpi_dQOld
    
#     # print(f"strain={strain}")
#     # print(f"strain_old={strain_old}")
#     # print(f"stress = {stress}")
#     print(f"W = {W}, F = {F}")
#     print(f"dqoi={qoi_derivatives}")

# eqps = internal_state_history[-1, Material.EQPS]
# dD_dY0_exact = eqps-(Y0+H*eqps)/(H+E)
# print(f"dqoi_dY0 = {qoi_derivatives:6e}")
# print(f"exact   = {dD_dY0_exact:6e}")

if __name__ == "__main__":
    E = 2.0
    nu = 0.25
    Y0 = 0.01
    H = E/50.0
    properties = {"elastic modulus": E,
                  "poisson ratio": nu,
                  "kinematics": "small deformations"}

    material = Material.create_material_model_functions(properties)
    compute_stress = jax.jit(jax.grad(material.compute_energy_density))

    strain_rate = 1e-3
    max_strain = 0.5
    steps = 50

    def compute_qoi(strain_new, isv_new, isv_old, dt, E, nu, Y0, H):
        """Compute increment of dissipation by right-side rectangle rule."""
        stress_new = compute_stress(strain_new, isv_old, dt, E, nu, Y0, H)
        plastic_strain_new = isv_new[Material.PLASTIC_STRAIN].reshape(3,3)
        plastic_strain_old = isv_old[Material.PLASTIC_STRAIN].reshape(3,3)
        return np.tensordot(stress_new, plastic_strain_new - plastic_strain_old)

    output, pi = simulate(material, max_strain, strain_rate, steps, compute_qoi, E, nu, Y0, H)

    pi_exact = Material.linear_hardening_potential(output[2][-1, Material.EQPS], Y0, H)

    print("===========SUMMARY===============")
    print(f"Plastic work: {pi:6e}")
    print(f"Exact:        {pi_exact:6e}")

    # print(f"strain hist {strain_history[:, 0, 0]}")
    # print(f"stress hist {stress_history[:, 0, 0]}")
    # plt.plot(strain_history[:, 0, 0], stress_history[:, 0, 0])
    # plt.show()
    