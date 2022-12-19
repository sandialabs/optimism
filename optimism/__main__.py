import argparse

from optimism.helper_methods.General import setup_dof_manager
from optimism.helper_methods.General import setup_essential_boundary_conditions
from optimism.helper_methods.General import setup_function_space
from optimism.helper_methods.General import setup_material_models
from optimism.helper_methods.General import setup_mesh
from optimism.helper_methods.General import setup_quadrature_rules

from optimism.helper_methods.Parser import dump_input_file
from optimism.helper_methods.Parser import parse_yaml_input_file


class MeshInputBlockError(Exception): pass
class EssentialBCBlockError(Exception): pass
class QuadratureBlockError(Exception): pass
class FunctionSpaceBlockError(Exception): pass


# TODO make some pretty print stuff, add authors, and emails, etc.
print('\nOptimism v0.0.1\n')

# create a parser
parser = argparse.ArgumentParser(
    prog='optimiSM',
    description='Rapid development platform for solid mechanics research using optimization tools')
parser.add_argument('-i', '--input_file',
                    help='File name of input file <input_file.yml>')

# parser the input arguments
args = parser.parse_args()
print('Input file = %s\n' % args.input_file)

# dump input file
dump_input_file(args.input_file)

print('######################################################################')
print('# BEGIN OPTIMISM LOGGING')
print('######################################################################\n')
# parse the input file
inputs = parse_yaml_input_file(args.input_file)

# setup a mesh
try:
    mesh = setup_mesh(inputs['mesh']['type'], inputs['mesh']['options'])
except KeyError:
    print('Error in input file mesh block.')
    print('Correct syntax is\n\nmesh:\n  type:    <mesh_type>\n  options: <dict_of_options>\n')
    raise MeshInputBlockError

# setup bcs
try:
    bcs = setup_essential_boundary_conditions(inputs['essential boundary conditions'])
except (AssertionError, KeyError, TypeError, ValueError):
    print('Error in input file essential boundary condition block.')
    print('Correct syntax is:\n\nessential boundary conditions:\n  - nodeset name: <str>\n    component:    <int>')
    print('  - nodeset name: <str>\n    component:    <int>\n...\n\n')
    raise EssentialBCBlockError

# setup quadrature rules
try:
    quad_rules = setup_quadrature_rules(inputs['quadrature rules'])
except (AttributeError, AssertionError, KeyError):
    print('Error in input file quadrature rules block.')
    print('Correct syntax is:\n\nquadrature rules:\n  - name:        <str>\n    cell degree: <int>\n    edge degree: <int>')
    print('  - name:        <str>\n    cell degree: <int>\n    edge degree: <int>\n  ...\n\n')
    raise QuadratureBlockError

# setup function spaces
try:
    f_space = setup_function_space(inputs['function space'], mesh, quad_rules)
except KeyError:
    print('Error in input file function space block.')
    print('Correct syntax is:\n\nfunction space:\n  quadrature rule: <str>\n\n')
    raise FunctionSpaceBlockError

# setup dof manager
dof_manager = setup_dof_manager(f_space, bcs, dim=2)

# setting up material models
mat_models = setup_material_models(inputs['material models'])

print('######################################################################')
print('# END OPTIMISM LOGGING')
print('######################################################################\n')