import abc
import dataclasses
import functools
import jax
import jax.numpy as np

from optimism import Interpolants
from optimism import Mesh
from optimism import QuadratureRule
from optimism import VTKWriter

class Field(abc.ABC):
    @abc.abstractmethod
    def interpolate(self, shape, U, el_id):
        pass

    @abc.abstractmethod
    def interpolate_gradient(self, dshape, U, el_id):
        pass

    @property
    @abc.abstractmethod
    def quadpoint_axis(self):
        pass

    @abc.abstractmethod
    def compute_shape_functions(self, points):
        pass

    @abc.abstractmethod
    def map_shape_functions(self, shapes, jacs):
        return shapes


class PkField(Field):
    """Standard Lagrange polynomial finite element fields."""
    quadpoint_axis = 0

    def __init__(self, k, mesh):
        # temporarily restrict to first order meshes.
        # Building the connectivity table for higher-order fields requires knowing the
        # simplex connectivity. The mesh object must be updated to store that.
        assert mesh.parentElement.degree == 1
        self.order = k
        self.element, self.element1d = Interpolants.make_parent_elements(k)
        self.mesh = mesh
        self.conns, self.coords = self._make_connectivity_and_coordinates()

    def interpolate(self, shape, field, el_id):
        e_vector = field[self.conns[el_id]]
        return shape.values@e_vector

    def interpolate_gradient(self, shape, field, el_id):
        return jax.vmap(lambda ev, dshp: np.tensordot(ev, dshp, (0, 0)), (None, 0))(field[self.conns[el_id]], shape.gradients)

    def compute_shape_functions(self, points):
        return Interpolants.compute_shapes(self.element, points)
    
    def map_shape_functions(self, shapes, jacs):
        shape_grads = jax.vmap(lambda dshp, J: dshp@np.linalg.inv(J))(shapes.gradients, jacs)
        return Interpolants.ShapeFunctions(shapes.values, shape_grads)
    
    def _make_connectivity_and_coordinates(self):
        if self.order == 1:
            return self.mesh.conns, self.mesh.coords
        
        conns = np.zeros((Mesh.num_elements(self.mesh), self.element.coordinates.shape[0]), dtype=int)

        # step 1/3: vertex nodes
        conns = conns.at[:, self.element.vertexNodes].set(self.mesh.conns)
        nodeOrdinalOffset = self.mesh.conns.max() + 1 # offset for later node numbering

        # The non-vertex nodes are placed using linear interpolation. When we add meshes that
        # have higher-order coordinate spaces, we must remember to update this part with
        # the higher-order interpolation.

        # step 2/3: mid-edge nodes (excluding vertices)
        edgeConns, edges = Mesh.create_edges(self.mesh.conns)
        A = np.column_stack((1.0 - self.element1d.coordinates[self.element1d.interiorNodes],
                             self.element1d.coordinates[self.element1d.interiorNodes]))
        edgeCoords = jax.vmap(lambda edgeConn: np.dot(A, self.mesh.coords[edgeConn, :]))(edgeConns)

        nNodesPerEdge = self.element1d.interiorNodes.size
        for e, edge in enumerate(edges):
            edgeNodeOrdinals = nodeOrdinalOffset + np.arange(e*nNodesPerEdge,(e+1)*nNodesPerEdge)
            
            elemLeft = edge[0]
            sideLeft = edge[1]
            edgeMasterNodes = self.element.faceNodes[sideLeft][self.element1d.interiorNodes]
            conns = conns.at[elemLeft, edgeMasterNodes].set(edgeNodeOrdinals)

            elemRight = edge[2]
            if elemRight >= 0:
                sideRight = edge[3]
                edgeMasterNodes = self.element.faceNodes[sideRight][self.element1d.interiorNodes]
                conns = conns.at[elemRight, edgeMasterNodes].set(np.flip(edgeNodeOrdinals))

        nEdges = edges.shape[0]
        nodeOrdinalOffset += nEdges*nNodesPerEdge # for offset of interior node numbering

        # step 3/3: interior nodes
        nInNodesPerTri = self.element.interiorNodes.shape[0]
        if nInNodesPerTri > 0:
            N0 = self.element.coordinates[self.element.interiorNodes, 0]
            N1 = self.element.coordinates[self.element.interiorNodes, 1]
            N2 = 1.0 - N0 - N1
            A = np.column_stack((N0, N1, N2))
            interiorCoords = jax.vmap(lambda triConn: np.dot(A, self.mesh.coords[triConn]))(self.mesh.conns)

            def add_element_interior_nodes(conn, newNodeOrdinals):
                return conn.at[self.element.interiorNodes].set(newNodeOrdinals)

            nTri = conns.shape[0]
            newNodeOrdinals = np.arange(nTri*nInNodesPerTri).reshape(nTri,nInNodesPerTri) \
                + nodeOrdinalOffset
            
            conns = jax.vmap(add_element_interior_nodes)(conns, newNodeOrdinals)
        else:
            interiorCoords = np.zeros((0, 2))
            
        coords = np.vstack((self.mesh.coords, edgeCoords.reshape(-1,2), interiorCoords.reshape(-1,2)))
        return conns, coords



class UniformField(Field):
    """A unique value for the whole mesh (things like time)."""
    quadpoint_axis = None

    def interpolate(self, shape, U, el_id):
        return U
    
    def interpolate_gradient(self, shape, U, el_id):
        raise NotImplementedError(f"Gradients not supported for {type(self).__name__}")
    
    def compute_shape_functions(self, points):
        return Interpolants.ShapeFunctions(np.array([]), np.array([]))
    
    def map_shape_functions(self, shapes, jacs):
        return super().map_shape_functions(shapes, jacs)


class QuadratureField(Field):
    """Arrays defined directly at quadrature points (things like internal variables)."""
    quadpoint_axis = 0

    def interpolate(self, shape, field, el_id):
        return field[el_id]
    
    def interpolate_gradient(self, shape, field, el_id):
        raise NotImplementedError(f"Gradients not supported for {type(self).__name__}")
    
    def compute_shape_functions(self, points):
        return Interpolants.ShapeFunctions(np.array([]), np.array([]))
    
    def map_shape_functions(self, shapes, jacs):
        return super().map_shape_functions(shapes, jacs)


@dataclasses.dataclass
class FieldInterpolation:
    """Abstract base class for specific types of field interpolations.

    This class should not be instantiated, only derived from.
    """
    field: int


class Value(FieldInterpolation):
    """Sentinel to indicate you want to interpolate the value of the field."""
    pass


class Gradient(FieldInterpolation):
    """Sentinel to indicate you want to interpolate the gradient of the field."""
    pass


def _choose_interpolation_function(input, spaces):
    """Helper function to choose correct field space interpolation function for each input."""
    if input.field >= len(spaces):
        raise IndexError(f"Field space {input.field} exceeds range, which is {len(spaces) - 1}.")
    if type(input) is Value:
        return spaces[input.field].interpolate
    elif type(input) is Gradient:
        return spaces[input.field].interpolate_gradient
    else:
        raise TypeError("Type of object in qfunction signature is invalid.")


class FieldEvaluator:
    def __init__(self, spaces: tuple[Field], qfunction_signature: tuple[FieldInterpolation], mesh: Mesh.Mesh, quadrature_rule: QuadratureRule.QuadratureRule) -> None:
        # the coord space should live on the Mesh eventually
        self._coord_space = PkField(mesh.parentElement.degree, mesh)
        self._coord_shapes = self._coord_space.compute_shape_functions(quadrature_rule.xigauss)

        self._spaces = spaces
        self._mesh = mesh
        self._quadrature_rule = quadrature_rule
        self._shapes = [space.compute_shape_functions(quadrature_rule.xigauss) for space in spaces]
        self._input_fields = tuple(input.field for input in qfunction_signature)
        self._interpolators = tuple(_choose_interpolation_function(input, self._spaces) for input in qfunction_signature)

    def evaluate(self, f, coords, *fields):
        f_vmap_axis = None
        elems_in_block = np.arange(Mesh.num_elements(self._mesh))
        compute_values = jax.vmap(self._evaluate_on_element, (f_vmap_axis, 0, None) + tuple(None for field in fields))
        return compute_values(f, elems_in_block, coords, *fields)
    
    def integrate(self, f, coords, *fields):
        f_vmap_axis = None
        elems_in_block = np.arange(Mesh.num_elements(self._mesh))
        integrate = jax.vmap(self._integrate_over_element, (f_vmap_axis, 0, None) + tuple(None for field in fields))
        return np.sum(integrate(f, elems_in_block, coords, *fields))
    
    def _evaluate_on_element(self, f, el_id, coords, *fields):
        jacs = self._coord_space.interpolate_gradient(self._coord_shapes, coords, el_id)
        shapes = [space.map_shape_functions(shape, jacs) for (space, shape) in zip(self._spaces, self._shapes)]
        f_args = [interp(shapes[field_id], fields[field_id], el_id) for (interp, field_id) in zip(self._interpolators, self._input_fields)]
        f_batch = jax.vmap(f, tuple(self._spaces[input].quadpoint_axis for input in self._input_fields))
        return f_batch(*f_args)
    
    def _integrate_over_element(self, f, el_id, coords, *fields):
        jacs = self._coord_space.interpolate_gradient(self._coord_shapes, coords, el_id)
        shapes = [space.map_shape_functions(shape, jacs) for (space, shape) in zip(self._spaces, self._shapes)]
        dVs = jax.vmap(lambda J, w: np.linalg.det(J)*w)(jacs, self._quadrature_rule.wgauss)
        f_args = [interp(shapes[field_id], fields[field_id], el_id) for (interp, field_id) in zip(self._interpolators, self._input_fields)]
        f_batch = jax.vmap(f, tuple(self._spaces[input].quadpoint_axis for input in self._input_fields))
        f_vals = f_batch(*f_args)
        return np.dot(f_vals, dVs)
