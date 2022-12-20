import argparse

from optimism.helper_methods.General import setup_blocks
from optimism.helper_methods.General import setup_dof_manager
from optimism.helper_methods.General import setup_essential_boundary_conditions
from optimism.helper_methods.General import setup_function_space
from optimism.helper_methods.General import setup_material_models
from optimism.helper_methods.General import setup_mechanics_functions
from optimism.helper_methods.General import setup_mesh
from optimism.helper_methods.General import setup_quadrature_rules

from optimism.helper_methods.QuasiStaticMechanics import run_quasi_static_mechanics_simulation
from optimism.helper_methods.StaticMechanics import run_static_mechanics_simulation
# from optimism.helper_methods.StaticMechanics import setup_mechanics_functions

from optimism.helper_methods.Parser import dump_input_file
from optimism.helper_methods.Parser import parse_yaml_input_file

from optimism.helper_methods.Postprocessor import setup_vtk_writer


class MeshInputBlockError(Exception): pass
class EssentialBCBlockError(Exception): pass
class QuadratureBlockError(Exception): pass
class FunctionSpaceBlockError(Exception): pass
class PhysicsError(Exception): pass
class TimeIntegrationError(Exception): pass


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
    bcs = setup_essential_boundary_conditions(inputs['boundary conditions'])
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

# TODO likely need some more error checking below

# setup dof manager
dof_manager = setup_dof_manager(f_space, bcs, dim=2)

# setting up material models
mat_models = setup_material_models(inputs['material models'])

# parse blocks: this simply switches out some names
if 'blocks' in inputs.keys():
    mat_models = setup_blocks(inputs['blocks'], mat_models)
else:
    print('WARNING: No block section. Each block must have a unique material model in this mode!')
    print('WARNING: The material model names should match block names in this mode!\n')

# post-processor
vtk_writer = setup_vtk_writer(inputs['postprocessor']['output file base name'], mesh)

# now switch on physics type
if inputs['physics'] == 'mechanics':
    # TODO currently only supported plane strain mode!
    # TODO make this parametric when you're not so lazy
    mech_functions = setup_mechanics_functions(f_space, mat_models)
    if inputs['time integration'] == 'static':
        run_static_mechanics_simulation(mesh, f_space, dof_manager, mech_functions, inputs['boundary conditions'],
                                        inputs['objective'], inputs['solver'], vtk_writer)
    elif inputs['time integration'] == 'quasi-static':
        run_quasi_static_mechanics_simulation(mesh, f_space, dof_manager, mech_functions, inputs['boundary conditions'],
                                              inputs['objective'], inputs['solver'], inputs['time'], inputs['postprocessor'])
    else:
        print('Unsupported time integration mode "%s"!' % input['time integration'])
        raise TimeIntegrationError
else:
    print('Unsupported physics mode "%s"!' % inputs['physics'])
    raise PhysicsError

print('######################################################################')
print('# END OPTIMISM LOGGING')
print('######################################################################\n')
