from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Mesh
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism.material import MaterialModelFactory

from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union


class MeshTypeError(Exception): pass
class MeshOptionsError(Exception): pass
class EssentialBCError(AssertionError): pass
class QuadratureError(AssertionError): pass


class QuadratureRulesContainer(NamedTuple):
    cell_quadrature_rule: QuadratureRule.QuadratureRule
    edge_quadrature_rule: QuadratureRule.QuadratureRule


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
    print('Finished setting up mesh.\n')
    return mesh


def setup_essential_boundary_conditions(bc_inputs: List[dict]) -> List[FunctionSpace.EssentialBC]:
    print('Setting up essential boundary conditions...')
    bcs = []
    # loop over bcs
    for n, bc in enumerate(bc_inputs['displacement']):
        try:
            assert 'nodeset' in bc.keys()
            assert 'component' in bc.keys()
            assert 'type' in bc.keys()
            print('    Nodeset   = %s' % bc['nodeset'])
            print('    Component = %s' % bc['component'])
            print('    Type      = %s' % bc['type'])
            print('    ...Ignoring other settings for now.')
            if n < len(bc_inputs['displacement']) - 1: print()
            bcs.append(FunctionSpace.EssentialBC(bc['nodeset'], int(bc['component'])))
        except AssertionError:
            print('\n\n')
            print('Error in bc %s' % bc)
            raise EssentialBCError

    print('Finished setting up essential boundary conditions.\n')
    return bcs


def setup_quadrature_rules(quadrature_inputs: dict) -> dict:
    print('Setting up quadrature rules...')
    quad_rules = {}
    n = 0
    for key, val in quadrature_inputs.items():
        try:
            assert 'cell degree' in val.keys()
            assert 'edge degree' in val.keys()
            print('    Name        = %s' % key)
            print('    Cell degree = %s' % val['cell degree'])
            print('    Edge degree = %s' % val['edge degree'])
            if n < len(quadrature_inputs) - 1: print()
            quad_rules[key] = QuadratureRulesContainer(
                QuadratureRule.create_quadrature_rule_on_triangle(val['cell degree']),
                QuadratureRule.create_quadrature_rule_1D(val['edge degree'])
            )
            n = n + 1
        except AssertionError:
            raise QuadratureError
    print('Finished setting up quadrature rules.\n')
    return quad_rules


def setup_function_space(
    f_space_inputs: dict, 
    mesh: Mesh.Mesh, 
    quad_rules: dict) -> FunctionSpace.FunctionSpace:
    
    print('Setting up function spaces...')
    print('    Quadrature rule = %s' % f_space_inputs['quadrature rule'])
    f_space = FunctionSpace.\
        construct_function_space(mesh, quad_rules[f_space_inputs['quadrature rule']].cell_quadrature_rule)
    print('Finished setting up function spaces.\n')
    return f_space


def setup_dof_manager(function_space: FunctionSpace.FunctionSpace,
                      essential_bcs: List[FunctionSpace.EssentialBC],
                      dim: Optional[int] = 2) -> FunctionSpace.DofManager:
    print('Setting up dof manager...')
    dof_manager = FunctionSpace.DofManager(function_space, dim, essential_bcs)
    print('Finished setting up dof manager.\n')
    return dof_manager


def setup_material_models(mat_model_inputs: dict) -> dict:
    print('Setting up material models...')
    mat_models = {}
    n = 0
    for key, val in mat_model_inputs.items():
        assert 'model name' in val.keys()
        assert 'model properties' in val.keys()
        print('    Name       = %s' % key)
        print('    Model name = %s' % val['model name'])
        print('    Model properties:')
        for sub_key, sub_val in val['model properties'].items():
            print('        %s = %s' % (sub_key, sub_val))
        if n < len(mat_model_inputs) - 1: print()
        mat_models[key] = MaterialModelFactory.\
            material_model_factory(val['model name'], val['model properties'])
        n = n + 1
    print('Finished setting up material models.\n')
    return mat_models


def setup_blocks(block_inputs: dict, mat_models: dict) -> dict:
    print('Setting up blocks...')
    return_dict = {}
    for n, block in enumerate(block_inputs):
        assert 'block name' in block.keys()
        assert 'material model' in block.keys()
        print('    Block name     = %s' % block['block name'])
        print('    Material model = %s' % block['material model'])
        if n < len(block_inputs) - 1: print()
        return_dict[block['block name']] = mat_models[block['material model']]
    print('Finished setting up blocks.\n')
    return return_dict


def setup_post_processor():
    pass