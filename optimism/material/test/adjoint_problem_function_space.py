from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mesh
from optimism.FunctionSpace import compute_element_volumes
from optimism.FunctionSpace import compute_element_volumes_axisymmetric
from optimism.FunctionSpace import map_element_shape_grads
import jax


def construct_function_space_for_adjoint(coords, mesh, quadratureRule, mode2D='cartesian'):

    shapeOnRef = Interpolants.compute_shapes(mesh.parentElement, quadratureRule.xigauss)

    shapes = jax.vmap(lambda elConns, elShape: elShape, (0, None))(mesh.conns, shapeOnRef.values)

    shapeGrads = jax.vmap(map_element_shape_grads, (None, 0, None, None))(coords, mesh.conns, mesh.parentElement, shapeOnRef.gradients)

    if mode2D == 'cartesian':
        el_vols = compute_element_volumes
    elif mode2D == 'axisymmetric':
        el_vols = compute_element_volumes_axisymmetric
    vols = jax.vmap(el_vols, (None, 0, None, 0, None))(coords, mesh.conns, mesh.parentElement, shapes, quadratureRule.wgauss)

    # unpack mesh and remake a mesh to make sure we get all the AD
    mesh = Mesh.Mesh(coords=coords, conns=mesh.conns, simplexNodesOrdinals=mesh.simplexNodesOrdinals,
                     parentElement=mesh.parentElement, parentElement1d=mesh.parentElement1d, blocks=mesh.blocks,
                     nodeSets=mesh.nodeSets, sideSets=mesh.sideSets)

    return FunctionSpace.FunctionSpace(shapes, vols, shapeGrads, mesh, quadratureRule)
