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
    def interpolate(self, shape, U, conn):
        pass

    @abc.abstractmethod
    def interpolate_gradient(self, dshape, U, conn):
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
        return shape@e_vector

    def interpolate_gradient(self, dshape, field, conn):
        return jax.vmap(lambda ev, dshp: np.tensordot(ev, dshp, (0, 0)), (None, 0))(field[conn], dshape)

    def compute_shape_functions(self, points):
        return Interpolants.compute_shapes(self.element, points)


class DummyShapeFunctions:
    values = None
    gradients = None


class UniformField(Field):
    """A unique value for the whole mesh (things like time)."""
    element_axis = None
    quadpoint_axis = None

    def interpolate(self, shape, field, conn):
        return field
    
    def interpolate_gradient(self, dshape, field, conn):
        grad_shape = field.shape + dshape.shape[-1]
        return np.zeros(grad_shape)
    
    def compute_shape_functions(self, points):
        return DummyShapeFunctions()


class QuadratureField(Field):
    """Arrays defined directly at quadrature points (things like internal variables)."""
    element_axis = 0
    quadpoint_axis = 0

    def __init__(self, dim):
        self.dim = dim

    def interpolate(self, shape, Q, conn):
        return Q
    
    def interpolate_gradient(self, dshape, Q, conn):
        raise NotImplementedError(f"Gradients not supported for {type(self).__name__}")
    
    def compute_shape_functions(self, points):
        return DummyShapeFunctions()


class ParametricElementField(Field):
    """Things like quadrature weights and shape function values."""
    element_axis = None
    quadpoint_axis = 0

    def interpolate(self, shape, field, conn):
        return field
    
    def interpolate_gradient(self, dshape, U, conn):
        raise NotImplementedError(f"Gradients not supported for {type(self).__name__}")
    
    def compute_shape_functions(self, points):
        return DummyShapeFunctions()

@dataclasses.dataclass
class Value:
    field: int


@dataclasses.dataclass
class Gradient:
    field: int


def pushforward(du_dxi, dX_dxi):
    return du_dxi@np.linalg.inv(dX_dxi)


def _make_interpolation_function(input, spaces, shapes):
    """Helper function to choose correct field space interpolation function for each input."""
    if input.field >= len(spaces):
        raise IndexError(f"Field space {input.field} exceeds number of field spaces, which is {len(spaces) - 1}.")
    if type(input) is Value:
        return functools.partial(spaces[input.field].interpolate, shapes[input.field].values)
    elif type(input) is Gradient:
        return functools.partial(spaces[input.field].interpolate_gradient, shapes[input.field].gradients)
    else:
        raise TypeError("Type of object in qfunction signature is invalid.")


class FieldEvaluator:
    def __init__(self, spaces, qfunction_signature, mesh, quadrature_rule):
        self._spaces = spaces
        self._mesh = mesh
        self._quadrature_rule = quadrature_rule
        self._shapes = [space.compute_shape_functions(quadrature_rule.xigauss) for space in spaces]
        self._input_fields = tuple(input.field for input in qfunction_signature)
        self._interpolators = tuple(_make_interpolation_function(input, self._spaces, self._shapes) for input in qfunction_signature)

    def evaluate(self, f, *fields):
        f_vmap_axis = None
        conns_vmap_axis = 0
        compute_values = jax.vmap(self._evaluate_on_element, (f_vmap_axis, conns_vmap_axis) + tuple(space.element_axis for space in self._spaces))
        return compute_values(f, self._mesh.conns, *fields)
    
    def _evaluate_on_element(self, f, el_conn, *fields):
        f_args = [interp(fields[field_id], el_conn) for (interp, field_id) in zip(self._interpolators, self._input_fields)]
        f_batch = jax.vmap(f, tuple(self._spaces[input].quadpoint_axis for input in self._input_fields))
        return f_batch(*f_args)


if __name__ == "__main__":
    p = 2
    dim = 2
    internal_var_dim = 1

    mesh = Mesh.construct_structured_mesh(2, 2, [0.0, 1.0], [0.0, 2.0], p)
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(2)
    
    u_space = PkField(p, dim)
    internal_variable_space = QuadratureField(internal_var_dim)
    time_space = UniformField()
    spaces = [u_space, internal_variable_space, time_space]

    inputs = (Value(0), Value(1), Value(2), Gradient(0))

    fs = FieldEvaluator([u_space, internal_variable_space, time_space], inputs, mesh, quad_rule)

    def f(u, Q, t):
        return np.dot(u, u)*Q[0]*t

    # print(f"f() = {f(np.array([1.0, 0.0]), Q[0, 0], 1.0)}")

    U = mesh.coords
    U = U.at[:, 1].set(0.0)
    Q = 2.0*np.ones((Mesh.num_elements(mesh), len(quad_rule), 1))
    t = 1.0

    # val = fs.evaluate(f, U, Q, t)
    # print(f"func vals shape = {val.shape}")
    # print(f"elements = {Mesh.num_elements(mesh)}, quad points = {len(quad_rule)}")
    # print(f"{val=}")

    # writer = VTKWriter.VTKWriter(mesh, "new_fs")
    # writer.write()

    def g(u, Q, t, du):
        return np.dot(u, u)*Q[0]*t*np.tensordot(du, du)
    
    test = fs._interpolators[0](U, mesh.conns[0])
    print(f"{test=}")

    print(fs._interpolators[1])
    test2 = fs._interpolators[1](Q, mesh.conns[0])
    print(f"{test2=}")
    
    val = fs.evaluate(g, U, Q, t)
    print(f"{val=}")

    def h(dX_dxi, du_dxi, w):
        E = 1.0
        nu = 0.0
        mu = 0.5*E/(1 + nu)
        lam = E*nu/(1 + nu)/(1 - 2*nu)
        dxi_dX = np.linalg.inv(dX_dxi)
        du_dX = du_dxi@dxi_dX
        dV = np.linalg.det(dX_dxi)*w
        strain = 0.5*(du_dX + du_dX.T)
        energy_density = mu*np.tensordot(strain, strain) + 0.5*lam*np.trace(strain)**2
        return energy_density*dV
    
    def area(dX_dxi, du_dxi, w):
        return np.linalg.det(dX_dxi)*w
    
    quad_weight_space = ParametricElementField()

    spaces = [u_space, u_space, quad_weight_space]
    qfunction_signature = [Gradient(0), Gradient(1), Value(2)]

    fs2 = FieldEvaluator(spaces, qfunction_signature, mesh, quad_rule)
    energies = fs2.evaluate(h, mesh.coords, U, quad_rule.wgauss)
    print(f"{energies=}")

    def potential(X, U):
        return np.sum(fs2.evaluate(h, X, U, quad_rule.wgauss))
    
    compute_force = jax.grad(potential, 1)
    R = compute_force(mesh.coords, U)
    print(f"{R=}")
