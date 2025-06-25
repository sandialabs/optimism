import abc
import jax
import jax.numpy as np

from optimism import Interpolants
from optimism import Mesh
from optimism import QuadratureRule
from optimism import VTKWriter

class Field(abc.ABC):
    pass

    @abc.abstractmethod
    def interpolate(self, U, shape, conn):
        pass

    @property
    @abc.abstractmethod
    def element_axis(self):
        pass

    @abc.abstractmethod
    def compute_shape_functions(self, points):
        pass

class FEField(Field):
    element_axis = None
    quadpoint_axis = 0

    def __init__(self, order, dim):
        self.order = order
        self.dim = dim
        self.element, self.element1d = Interpolants.make_parent_elements(order)

    def interpolate(self, field, shape, conn):
        e_vector = field[conn]
        return shape@e_vector

    def interpolate_gradient(self, field, dshape, conn):
        return jax.vmap(lambda ev, dshp: np.tensordot(ev, dshp, (0, 0)), (None, 0))(field[conn], dshape)

    def compute_shape_functions(self, points):
        return Interpolants.compute_shapes(self.element, points)


class DummyShapeFunctions:
    values = None
    gradients = None


class UniformScalarField(Field):
    element_axis = None
    quadpoint_axis = None

    def interpolate(self, field, shape, conn):
        return field
    
    def compute_shape_functions(self, points):
        return DummyShapeFunctions()


class QuadratureField(Field):
    element_axis = 0
    quadpoint_axis = 0

    def __init__(self, dim):
        self.dim = dim

    def interpolate(self, Q, shape, conn):
        return Q
    
    def compute_shape_functions(self, points):
        return DummyShapeFunctions()

class FunctionSpace2:
    def __init__(self, spaces, mesh, quadrature_rule):
        self._spaces = spaces
        self._mesh = mesh
        self._quadrature_rule = quadrature_rule
        self._shapes = [space.compute_shape_functions(quadrature_rule.xigauss) for space in spaces]

    def evaluate(self, f, *fields):
        f_vmap_axis = None
        conns_vmap_axis = 0
        compute_values = jax.vmap(self._evaluate_on_element, (f_vmap_axis, conns_vmap_axis) + tuple(space.element_axis for space in self._spaces))
        return compute_values(f, self._mesh.conns, *fields)
    
    def _evaluate_on_element(self, f, el_conn, *fields):
        f_args = [z[0].interpolate(z[1], z[2].values, el_conn) for z in zip(self._spaces, fields, self._shapes)]
        f_args += [self._spaces[0].interpolate_gradient(fields[0], self._shapes[0].gradients, el_conn)]
        f_batch = jax.vmap(f, tuple(space.quadpoint_axis for space in self._spaces) + (0,))
        return f_batch(*f_args)


if __name__ == "__main__":
    p = 1
    dim = 2
    internal_var_dim = 1

    mesh = Mesh.construct_structured_mesh(2, 2, [0.0, 1.0], [0.0, 2.0], p)
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(2)
    
    u_space = FEField(p, dim)
    internal_variable_space = QuadratureField(internal_var_dim)
    time_space = UniformScalarField()
    spaces = [u_space, internal_variable_space, time_space]

    fs = FunctionSpace2([u_space, internal_variable_space, time_space], mesh, quad_rule)

    def f(u, Q, t):
        return np.dot(u, u)*Q[0]*t
    
    U = jax.vmap(lambda X: np.array([X[0], 0.0]))(mesh.coords)
    print(f"{U=}")
    Q = 2.0*np.ones((Mesh.num_elements(mesh), len(quad_rule), 1))
    t = 1.0

    # print(f"f() = {f(np.array([1.0, 0.0]), Q[0, 0], 1.0)}")

    # val = fs.evaluate(f, U, Q, t)
    # print(f"func vals shape = {val.shape}")
    # print(f"elements = {Mesh.num_elements(mesh)}, quad points = {len(quad_rule)}")
    # print(f"{val=}")

    # writer = VTKWriter.VTKWriter(mesh, "new_fs")
    # writer.write()

    def g(u, Q, t, du):
        jax.debug.print("du={du}", du=du)
        return np.tensordot(du, du)
    
    val = fs.evaluate(g, U, Q, t)
    print(f"{val=}")
