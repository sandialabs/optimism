from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Mesh


def create_current_mesh(mesh, designParams):
    return Mesh.mesh_with_coords(mesh, designParams)


def create_function_space(mesh, quadRule, designParams):
    fs = FunctionSpace.construct_weighted_function_space(create_current_mesh(mesh, designParams), quadRule)
    return fs


@partial(jit, static_argnums=(2,))
def create_parameters_from_design_vars(coordsu, mesh, dofManager):
    return dofManager.create_field(coordsu, dofManager.get_bc_values(mesh.coords))


def create_initial_design_vars(mesh, dofManager, quadRule):
    return dofManager.get_unknown_values(mesh.coords)
