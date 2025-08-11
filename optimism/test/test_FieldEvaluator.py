import pytest
import jax.numpy as np

from optimism import Mesh
from optimism.FieldEvaluator import *

class TestBasics:
    coord_degree = 1
    dim = 2
    length = 3.0
    height = 2.0
    mesh = Mesh.construct_structured_mesh(2, 2, [0.0, length], [0.0, height], coord_degree)
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(4)

    def test_gradient_evaluation(self):
        "Check the gradient of an affine field"
        p = 1
        spaces = PkField(p, self.mesh),
        inputs = Gradient(0),
        field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)

        target_disp_grad = np.array([[0.1, 0.01],
                                     [0.05, 0.3]])
        #coords_linear = self.mesh.coords[self.mesh.simplexNodesOrdinals, :]
        U = np.einsum('aj, ij', self.mesh.coords, target_disp_grad + np.identity(2)) - self.mesh.coords

        def f(dudX):
            return dudX
        
        disp_grads = field_evaluator.evaluate(f, self.mesh.coords, U)

        for H in disp_grads.reshape(-1, 2, 2):
            assert pytest.approx(H) == target_disp_grad

    def test_trivial_integral(self):
        spaces = PkField(self.coord_degree, self.mesh),
        inputs = Value(0),
        field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)
        U = np.zeros_like(self.mesh.coords)
        def f(u):
            return 1.0
        area = field_evaluator.integrate(f, self.mesh.coords, U)
        assert pytest.approx(area) == self.length*self.height
    
    def test_integral_with_one_nodal_field(self):
        "Computes area in a non-trivial way, checking consistency of gradient and integral operators."
        spaces = PkField(self.coord_degree, self.mesh),
        POSITION = 0
        # We're taking the gradient of position, which is just the identity tensor
        inputs = Gradient(POSITION),
        field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)

        def f(dXdX):
            # note dXdX == identity, so
            # trace(dXdX)/dim = 1
            return np.trace(dXdX)/self.dim
        
        area = field_evaluator.integrate(f, self.mesh.coords, self.mesh.coords)
        assert pytest.approx(area) == self.length*self.height

    def test_helmholtz(self):
        spaces = PkField(2, self.mesh), QuadratureField()
        
        inputs = Value(0), Gradient(0), Value(1)
        field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)
        
        def f(u, dudX, q):
            return 0.5*q[0]*(u*u + np.dot(dudX, dudX))
        
        # u(X, Y) = 0.1*X + 0.01*Y + 2
        target_grad = np.array([0.1, 0.01])
        U = spaces[0].coords@target_grad + 2.0
        
        Q = 2*np.ones((Mesh.num_elements(self.mesh), len(self.quad_rule), 1))
        
        energy = field_evaluator.integrate(f, self.mesh.coords, U, Q)
        print(f"{energy:.12e}")
        assert energy == pytest.approx(28.0994)

    def test_nonexistent_field_id_gets_error(self):
        spaces = PkField(self.coord_degree, self.mesh),
        inputs = Gradient(1), # there is no field 1
        with pytest.raises(AssertionError):
            field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)

