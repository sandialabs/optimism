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
    pass

    @abc.abstractmethod
    def interpolate(self, shape, U, conn, jac):
        pass

    @abc.abstractmethod
    def interpolate_gradient(self, dshape, U, conn, jac):
        pass

    @property
    @abc.abstractmethod
    def element_axis(self):
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
    element_axis = None
    quadpoint_axis = 0

    def __init__(self, k, dim):
        self.order = k
        self.dim = dim
        self.element, self.element1d = Interpolants.make_parent_elements(k)

    def interpolate(self, shape, field, conn):
        e_vector = field[conn]
        return shape.values@e_vector

    def interpolate_gradient(self, shape, field, conn):
        return jax.vmap(lambda ev, dshp: np.tensordot(ev, dshp, (0, 0)), (None, 0))(field[conn], shape.gradients)

    def compute_shape_functions(self, points):
        return Interpolants.compute_shapes(self.element, points)
    
    def map_shape_functions(self, shapes, jacs):
        shape_grads = jax.vmap(lambda dshp, J: dshp@np.linalg.inv(J))(shapes.gradients, jacs)
        return Interpolants.ShapeFunctions(shapes.values, shape_grads)


class UniformField(Field):
    """A unique value for the whole mesh (things like time)."""
    element_axis = None
    quadpoint_axis = None

    def interpolate(self, shape, field, conn):
        return field
    
    def interpolate_gradient(self, shape, field, conn):
        raise NotImplementedError(f"Gradients not supported for {type(self).__name__}")
    
    def compute_shape_functions(self, points):
        return Interpolants.ShapeFunctions(np.array([]), np.array([]))
    
    def map_shape_functions(self, shapes, jacs):
        return super().map_shape_functions(shapes, jacs)


class QuadratureField(Field):
    """Arrays defined directly at quadrature points (things like internal variables)."""
    element_axis = 0
    quadpoint_axis = 0

    def __init__(self, dim):
        self.dim = dim

    def interpolate(self, shape, field, conn):
        return field
    
    def interpolate_gradient(self, shape, field, conn):
        raise NotImplementedError(f"Gradients not supported for {type(self).__name__}")
    
    def compute_shape_functions(self, points):
        return Interpolants.ShapeFunctions(np.array([]), np.array([]))
    
    def map_shape_functions(self, shapes, jacs):
        return super().map_shape_functions(shapes, jacs)


@dataclasses.dataclass
class Value:
    """Sentinel to indicate you want to interpolate the value of the field."""
    field: int


@dataclasses.dataclass
class Gradient:
    """Sentinel to indicate you want to interpolate the gradient of the field."""
    field: int


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
    def __init__(self, spaces, qfunction_signature, mesh, quadrature_rule):
        # the coord space should live on the Mesh
        self._coord_space = PkField(mesh.parentElement.degree, mesh.coords.shape[1])
        self._coord_shapes = self._coord_space.compute_shape_functions(quadrature_rule.xigauss)

        self._spaces = spaces
        self._mesh = mesh
        self._quadrature_rule = quadrature_rule
        self._shapes = [space.compute_shape_functions(quadrature_rule.xigauss) for space in spaces]
        self._input_fields = tuple(input.field for input in qfunction_signature)
        self._interpolators = tuple(_choose_interpolation_function(input, self._spaces) for input in qfunction_signature)

    def evaluate(self, f, coords, *fields):
        f_vmap_axis = None
        conns_vmap_axis = 0
        coords_vmap_axis = None
        compute_values = jax.vmap(self._evaluate_on_element, (f_vmap_axis, conns_vmap_axis, coords_vmap_axis) + tuple(space.element_axis for space in self._spaces))
        return compute_values(f, self._mesh.conns, coords, *fields)
    
    def _evaluate_on_element(self, f, el_conn, coords, *fields):
        jacs = self._coord_space.interpolate_gradient(self._coord_shapes, coords, el_conn)
        shapes = [space.map_shape_functions(shape, jacs) for (space, shape) in zip(self._spaces, self._shapes)]
        f_args = [interp(shapes[field_id], fields[field_id], el_conn) for (interp, field_id) in zip(self._interpolators, self._input_fields)]
        f_batch = jax.vmap(f, tuple(self._spaces[input].quadpoint_axis for input in self._input_fields))
        return f_batch(*f_args)
    
    def _integrate_over_element(self, f, el_conn, coords, *fields):
        jacs = self._coord_space.interpolate_gradient(self._coord_shapes, coords, el_conn)
        shapes = [space.map_shape_functions(shape, jacs) for (space, shape) in zip(self._spaces, self._shapes)]
        dVs = jax.vmap(lambda J, w: np.linalg.det(J)*w)(jacs, self._quadrature_rule.wgauss)
        f_args = [interp(shapes[field_id], fields[field_id], el_conn) for (interp, field_id) in zip(self._interpolators, self._input_fields)]
        f_batch = jax.vmap(f, tuple(self._spaces[input].quadpoint_axis for input in self._input_fields))
        f_vals = f_batch(*f_args)
        return np.dot(f_vals, dVs)
    
    def integrate(self, f, coords, *fields):
        f_vmap_axis = None
        conns_vmap_axis = 0
        coords_vmap_axis = None
        compute_values = jax.vmap(self._integrate_over_element, (f_vmap_axis, conns_vmap_axis, coords_vmap_axis) + tuple(space.element_axis for space in self._spaces))
        return np.sum(compute_values(f, self._mesh.conns, coords, *fields))
