from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Mesh
from optimism import QuadratureRule
from optimism import ReadExodusMesh

from typing import Union


class MeshTypeError(Exception): pass
class MeshOptionsError(Exception): pass


def setup_mesh(mesh_type: str, mesh_options: Union[None, dict]) -> Mesh.Mesh:
    print('Setting up mesh...')
    try:
        if mesh_type == 'structured mesh':
            print('    Mesh type is "structured mesh"')
            try:
                print('    Nx      = %s' % mesh_options['Nx'])
                print('    Ny      = %s' % mesh_options['Ny'])
                print('    xExtent = %s' % mesh_options['xExtent'])
                print('    yExtent = %s' % mesh_options['yExtent'])
                mesh = Mesh.construct_structured_mesh(mesh_options['Nx'], mesh_options['Ny'],
                                                    mesh_options['xExtent'], mesh_options['yExtent'])
            except (IndexError, KeyError, TypeError):
                print('\n\n')
                print('Error parsing structured mesh inputs')
                print('For a mesh of type "structured mesh" the input syntax is the following:\n')
                print('mesh:\n  type: structured mesh\n  options:')
                print('    Nx: <int>\n    Ny: <int>\n    xExtent: <list<float,float>>\n    yExtent: <list<float,float>>\n\n')
                raise MeshOptionsError
        elif mesh_type == 'exodus mesh': 
            print('    Mesh type is "exodus mesh"')
            try:
                print('    File name = %s' % mesh_options['file'])
                mesh = ReadExodusMesh.read_exodus_mesh(mesh_options['file'])
            except FileNotFoundError:
                print('\n\n')
                print('Could not read exodus mesh file = %s\n\n' % mesh_options['file'])
                raise MeshOptionsError
            except KeyError:
                print('\n\n')
                print('Error parsing exodus mesh inputs')
                print('For a mesh of type "exodus mesh" the input syntax is the following:\n')
                print('mesh:\n  type: exodus mesh\n  options:\n    file: <mesh_file.e>\n\n')
                raise MeshOptionsError
        else:
            raise MeshTypeError
        print('Setup mesh.\n')
    except ValueError:
        print('Mesh type needs to be "structured mesh" or "exodus mesh"')
    return mesh


def setup_essential_boundary_conditions():
    pass


def setup_quadrature_rules():
    pass


def setup_function_spaces():
    pass


def setup_post_processor():
    pass