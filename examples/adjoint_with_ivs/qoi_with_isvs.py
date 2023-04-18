from functools import partial
import numpy as onp
import jax
from jax import numpy as np
import matplotlib.pyplot as plt

import parameterized_j2 as Material
import nonlinear


@partial(jax.custom_vjp, nondiff_argnums=(0, 4))
def simulate(material, max_strain, strain_rate, steps, compute_qoi, *params):
    pi, _ = simulate_fwd(material, max_strain, strain_rate, steps, compute_qoi, *params)
    return pi


def simulate_fwd(material, max_strain, strain_rate, steps, compute_qoi, *params):
    t_max = max_strain/strain_rate
    dt = t_max/steps
    compute_stress = jax.jit(jax.grad(material.compute_energy_density))
    compute_tangents = jax.jit(jax.jacfwd(compute_stress))
    compute_state_new = jax.jit(material.compute_state_new)
    #compute_energy_density = jax.jit(material.compute_energy_density)

    def r(lateral_strain, axial_strain, state, dt, *params):
        """Compute residual for uniaxial stress state."""
        strain = np.diag(np.array([axial_strain, lateral_strain, lateral_strain]))
        stress = compute_stress(strain, state, dt, *params)
        return stress[1,1]

    dr = jax.jacfwd(r)
    solve = jax.jit(partial(nonlinear.solve, r, dr))

    t = 0.0
    strain = np.zeros((3,3))
    internal_state = material.compute_initial_state()
    pi = 0.0
    stress_history = onp.zeros((steps, 3, 3))
    strain_history = onp.zeros((steps, 3, 3))
    internal_state_history = onp.zeros((steps, internal_state.shape[0]))
    internal_state_history[0] = internal_state
    time_history = onp.zeros(steps)

    # Phase 1: Simulation
    for i in range(1, steps):
        print("-----------------------")
        print(f"Step {i}")
        strain_old = strain
        strain = strain.at[0,0].add(strain_rate*dt)
        t += dt

        lat_strain = solve(strain[1,1], strain[0,0], internal_state, dt, *params)

        # post-process and save data needed for reverse pass
        strain = np.diag(np.array([strain[0,0], lat_strain, lat_strain]))
        stress = compute_stress(strain, internal_state, dt, *params)
        internal_state_old = internal_state
        internal_state = compute_state_new(strain, internal_state, dt, *params)
        pi += compute_qoi(strain, internal_state, internal_state_old, dt, *params)
        strain_history[i] = strain
        stress_history[i] = stress
        internal_state_history[i] = internal_state
        time_history[i] = t

        print(f"residual norm: {np.abs(stress[1,1])}")
        print(f"strain = {strain[0,0]}, \tstress = {stress[0,0]}")
        print(f"internal={internal_state}")
        print(f"QOI: {pi:6e}")
        
        history = strain_history, stress_history, internal_state_history, time_history

    return pi, (history, params)


def simulate_bwd(material, compute_qoi, tape, cotangent):
    history, params = tape
    strain_history, stress_history, internal_state_history, time_history = history
    compute_stress = jax.jit(jax.grad(material.compute_energy_density))
    compute_state_new = jax.jit(material.compute_state_new)
    def r(lateral_strain, axial_strain, state, dt, *params):
        """Compute residual for uniaxial stress state."""
        strain = np.diag(np.array([axial_strain, lateral_strain, lateral_strain]))
        stress = compute_stress(strain, state, dt, *params)
        return stress[1,1]
    dr = jax.jit(jax.jacfwd(r))

    compute_dpi = jax.jit(jax.grad(compute_qoi, (0, 1, 2, 4)))
    compute_dq = jax.jit(jax.jacrev(compute_state_new, (0, 1, 3)))
    compute_dsigma = jax.jit(jax.jacrev(compute_stress, (0, 1, 3)))
    
    qoi_derivatives = np.zeros_like(*params)
    adjoint_internal_state_force = np.zeros_like(internal_state_history[0])
    F = 0.0 # adjoint load
    print("\n\n")
    print("ADJOINT PHASE")
    for i in reversed(range(1, steps)):
        print("-------------------------")
        print(f"Step {i}")
        strain = strain_history[i]
        strain_old = strain_history[i-1]
        stress = stress_history[i]
        internal_state = internal_state_history[i]
        internal_state_old = internal_state_history[i-1]
        dt = time_history[i] - time_history[i-1]
        
        dpi_dH, dpi_dQ, dpi_dQOld, dpi_dp = compute_dpi(strain, internal_state, internal_state_old, dt, *params)
        dQ_dH, dQ_dQOld, dQ_dp = compute_dq(strain, internal_state_old, dt, *params)
        K = dr(strain[1,1], strain[0,0], internal_state_old, dt, *params)
        F = -dpi_dH[1,1]
        adjoint_internal_state_force += dpi_dQ
        F -= np.tensordot(adjoint_internal_state_force, dQ_dH, axes=1)[1,1]
        W = F/K

        qoi_derivatives += dpi_dp
        dSigma_dH, dSigma_dQOld, dSigma_dp = compute_dsigma(strain, internal_state_old, dt, *params)
        qoi_derivatives += W*dSigma_dp[1,1]
        qoi_derivatives += np.dot(adjoint_internal_state_force, dQ_dp)

        adjoint_internal_state_force = np.tensordot(adjoint_internal_state_force, dQ_dQOld, axes=1)
        adjoint_internal_state_force += W*dSigma_dQOld[1,1]
        adjoint_internal_state_force += dpi_dQOld
        
        # print(f"strain={strain}")
        # print(f"strain_old={strain_old}")
        # print(f"stress = {stress}")
        print(f"W = {W}, F = {F}")
        print(f"adjoint internal var force = {adjoint_internal_state_force}")
        print(f"dqoi={qoi_derivatives}")
        
    return 0.0, 0.0, 0, cotangent*qoi_derivatives

simulate.defvjp(simulate_fwd, simulate_bwd)

if __name__ == "__main__":
    E = 2.0
    nu = 0.25
    Y0 = 0.01
    H = E/50.0
    params = np.array([E, nu, Y0, H])
    properties = {"kinematics": "small deformations"}

    material = Material.create_material_model_functions(properties)
    compute_stress = jax.jit(jax.grad(material.compute_energy_density))

    strain_rate = 1e-3
    max_strain = 0.5
    steps = 50

    def compute_qoi(strain_new, isv_new, isv_old, dt, params):
        """Compute increment of dissipation by right-side rectangle rule."""
        stress_new = compute_stress(strain_new, isv_old, dt, params)
        plastic_strain_new = isv_new[Material.STATE_PLASTIC_STRAIN].reshape(3,3)
        plastic_strain_old = isv_old[Material.STATE_PLASTIC_STRAIN].reshape(3,3)
        return np.tensordot(stress_new, plastic_strain_new - plastic_strain_old)

    pi, dpi_dp = jax.value_and_grad(simulate, 5)(material, max_strain, strain_rate, steps, compute_qoi, params)

    eqps = (E*max_strain - Y0)/(E + H)
    pi_exact = Y0*eqps + 0.5*H*eqps**2

    print("===========SUMMARY===============")
    print(f"Plastic work: {pi:6e}")
    print(f"Exact:        {pi_exact:6e}")

    # print(f"strain hist {strain_history[:, 0, 0]}")
    # print(f"stress hist {stress_history[:, 0, 0]}")
    # plt.plot(strain_history[:, 0, 0], stress_history[:, 0, 0])
    # plt.show()
    
    print(f"dpi_dE = {dpi_dp[0]:6e}")
    print(f"exact  = {(Y0+H*eqps)*(H*max_strain+Y0)/(H+E)**2:6e}")
    dD_dY0_exact = eqps - (Y0+H*eqps)/(H+E)
    print(f"dpi_dY0  = {dpi_dp[2]:6e}")
    print(f"exact    = {dD_dY0_exact:6e}")
    dpi_dH_exact = -(Y0+H*eqps)*(E*max_strain-Y0)/(H+E)**2
    print(f"dpi_dH = {dpi_dp[3]:6e}")
    print(f"exact = {dpi_dH_exact:6e}")