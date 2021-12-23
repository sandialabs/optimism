from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import QuadratureRule


#expecting the quadrature densities to be in p[2]


def create_current_mesh(mesh, designParams):
    return mesh


def create_function_space(mesh, quadRule, designParams):
    return FunctionSpace.construct_weighted_function_space(mesh,
                                                           quadRule,
                                                           quadratureWeights=designParams)


def create_parameters_from_design_vars(chi, mesh, dofManager):
    return compute_phases(chi)


def create_initial_design_vars(mesh, dofManager, quadRule):
    return 1.0 * np.ones( (mesh.conns.shape[0], QuadratureRule.len(quadRule)) )


### private ###


def ramp(rho):
    eps0 = 1e-7
    num = (1.0-eps0) * rho
    denom =  1.0 + 8.0 * (1.0-rho)
    return eps0 + num / denom


def compute_phases(chi):
    rho = 1.0 / ( 1.0 + np.exp(-chi) )
    return ramp(rho)

