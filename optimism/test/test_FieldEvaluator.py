import pytest
import jax.numpy as np

from optimism import Mesh
from optimism.FieldEvaluator import *

class TestBasics:
    p = 2
    dim = 2
    mesh = Mesh.construct_structured_mesh(2, 2, [0.0, 3.0], [0.0, 2.0], p)
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(2)

    def test_area(self):
        spaces = [PkField(self.p, self.dim)]
        inputs = [Value(0)]
        field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)
        U = np.zeros_like(self.mesh.coords)
        def f(u):
            return 1.0
        area = field_evaluator.integrate(f, self.mesh.coords, U)
        assert pytest.approx(area) == 6
    
    def test_area_another_way(self):
        spaces = [PkField(self.p, self.dim)]
        inputs = [Gradient(0)]
        field_evaluator = FieldEvaluator(spaces, inputs, self.mesh, self.quad_rule)
        
        def f(dXdX):
            return np.trace(dXdX)/self.dim
        
        area = field_evaluator.integrate(f, self.mesh.coords, self.mesh.coords)
        assert pytest.approx(area) == 6
    
    def test_gradient_interpolation(self):
        p = 2
        spaces = [PkField(p, self.dim)]
        inputs = [Gradient(0)]
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

