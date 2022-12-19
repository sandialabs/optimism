from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Mesh
from optimism import QuadratureRule
from optimism import ReadExodusMesh

from typing import List
from typing import Tuple
from typing import Union


class MeshTypeError(Exception): pass
class MeshOptionsError(Exception): pass
class EssentialBCError(AssertionError): pass
class QuadratureError(AssertionError): pass


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
    except ValueError:
        print('Mesh type needs to be "structured mesh" or "exodus mesh"')
    print('Setup mesh.\n')
    return mesh


def setup_essential_boundary_conditions(bc_inputs: List[dict]) -> List[FunctionSpace.EssentialBC]:
    print('Setting up essential boundary conditions...')
    bcs = []
    # loop over bcs
    for n, bc in enumerate(bc_inputs):
        try:
            assert 'nodeset name' in bc.keys()
            assert 'component' in bc.keys()
            print('    Nodeset name = %s' % bc['nodeset name'])
            print('    Component    = %s' % bc['component'])
            if n < len(bc_inputs) - 1: print()
            bcs.append(FunctionSpace.EssentialBC(bc['nodeset name'], int(bc['component'])))
        except AssertionError:
            print('\n\n')
            print('Error in bc %s' % bc)
            raise EssentialBCError

    print('Setup essential boundary conditions\n')
    return bcs


def setup_quadrature_rules(quadrature_inputs: dict) -> Tuple[QuadratureRule.QuadratureRule, QuadratureRule.QuadratureRule]:
    print('Setting up quadrature rules...')
    try:
        assert 'cell degree' in quadrature_inputs.keys()
        assert 'edge degree' in quadrature_inputs.keys()
        print('    Cell degree = %s' % quadrature_inputs['cell degree'])
        print('    Edge degree = %s' % quadrature_inputs['edge degree'])
        cell_degree = quadrature_inputs['cell degree']
        edge_degree = quadrature_inputs['edge degree']
    except AssertionError:
        raise QuadratureError
    cell_q_rule = QuadratureRule.create_quadrature_rule_on_triangle(cell_degree)
    edge_q_rule = QuadratureRule.create_quadrature_rule_1D(edge_degree)
    print('Setup quadrature rules.\n')
    return cell_q_rule, edge_q_rule


def setup_function_spaces():
    print('Setting up function spaces...')


    print('Setup function spaces.\n')


def setup_post_processor():
    pass