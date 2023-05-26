import jax.numpy as np

from optimism import FunctionSpace


def compute_traction_potential_energy(fs, U, quadRule, edges, load, time=0.0):
    def compute_energy_density(u, X, n, t):
        traction = load(X, n, t)
        return -np.dot(u, traction)
    return FunctionSpace.integrate_function_on_edges(fs, compute_energy_density, U, quadRule, edges, time)
