from optimism.JaxConfig import *
from optimism.Parser import *

from optimism.FunctionSpace import construct_function_space
from optimism.FunctionSpace import DofManager
from optimism.Mechanics import create_mechanics_functions
from optimism.ReadExodusMesh import read_exodus_mesh
from optimism.QuadratureRule import create_quadrature_rule_on_triangle

if __name__ == '__main__':
    inputs = parse_input_file('./lattice_geometry.yaml')
    mesh = read_exodus_mesh(parse_mesh(inputs))
    cell_quad_rule = create_quadrature_rule_on_triangle(parse_quadrature_rule_cell(inputs))
    function_space = construct_function_space(mesh, cell_quad_rule, parse_function_space(inputs))
    dof_manager = DofManager(function_space, 2, parse_boundary_conditions(inputs))
    material_model, props = parse_material_model(inputs)

    if material_model == 'Neohookean':
        from optimism.material import Neohookean
        material_model = Neohookean.create_material_model_functions(props)
    else:
        assert False, 'Implement this'
    
    mech_functions = create_mechanics_functions(function_space, 'plane strain', material_model)