import argparse

from optimism.helper_methods.General import setup_mesh
from optimism.helper_methods.Parser import parse_yaml_input_file


class MeshInputBlockError(Exception): pass


# TODO make some pretty print stuff
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

# parse the input file
inputs = parse_yaml_input_file(args.input_file)

# setup a mesh
try:
    mesh = setup_mesh(inputs['mesh']['type'], inputs['mesh']['options'])
except KeyError:
    print('Error in input file mesh block.')
    print('Correct syntax is\n\nmesh:\n  type:    <mesh_type>\n  options: <dict_of_options>\n')
    raise MeshInputBlockError

