import pytest
import jax.numpy as np

from optimism import Mesh
from optimism.FieldOperator import *

class TestFieldOperator:
    coord_degree = 1
    dim = 2
    length = 3.0
    height = 2.0
    mesh = Mesh.construct_structured_mesh(2, 2, [0.0, length], [0.0, height], coord_degree)
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(4)

    def test_gradient_evaluation(self):
        "Check the gradient of an affine field"
        k = 1
        spaces = PkField(k, self.mesh),
        integrand_signature = Gradient(0),
        field_operator = FieldOperator(spaces, integrand_signature, self.mesh, self.quad_rule)

        target_disp_grad = np.array([[0.1, 0.01],
                                     [0.05, 0.3]])
        U = np.einsum('aj, ij', self.mesh.coords, target_disp_grad + np.identity(2)) - self.mesh.coords

        def f(dudX):
            return dudX

        disp_grads = field_operator.evaluate(f, self.mesh.coords, self.mesh.blocks['block_0'], U)

        for H in disp_grads.reshape(-1, 2, 2):
            assert pytest.approx(H) == target_disp_grad

    def test_trivial_integral(self):
        spaces = PkField(self.coord_degree, self.mesh),
        integrand_signature = Value(0),
        field_operator = FieldOperator(spaces, integrand_signature, self.mesh, self.quad_rule)
        U = np.zeros_like(self.mesh.coords)
        def f(u):
            return 1.0
        area = field_operator.integrate(f, self.mesh.coords, self.mesh.blocks['block_0'], U)
        assert pytest.approx(area) == self.length*self.height
    
    def test_integral_with_one_nodal_field(self):
        "Computes area in a non-trivial way, checking consistency of gradient interpolation."
        integrand_signature = PkField(self.coord_degree, self.mesh),
        POSITION = 0
        # We're taking the gradient of position, which is just the identity tensor
        inputs = Gradient(POSITION),
        field_operator = FieldOperator(integrand_signature, inputs, self.mesh, self.quad_rule)

        def f(dXdX):
            # note dXdX == identity, so
            # trace(dXdX)/dim = 1
            return np.trace(dXdX)/self.dim
        
        area = field_operator.integrate(f, self.mesh.coords, self.mesh.blocks['block_0'], self.mesh.coords)
        assert pytest.approx(area) == self.length*self.height

    def test_helmholtz(self):
        "Tests interpolation, gradient, and simple use of a QuadratureField"
        spaces = PkField(2, self.mesh), QuadratureField()
        
        integrand_signature = Value(0), Gradient(0), Value(1)
        field_operator = FieldOperator(spaces, integrand_signature, self.mesh, self.quad_rule)
        
        def f(u, dudX, q):
            return 0.5*q[0]*(u*u + np.dot(dudX, dudX))
        
        # u(X, Y) = 0.1*X + 0.01*Y + 2
        target_grad = np.array([0.1, 0.01])
        U = spaces[0].coords@target_grad + 2.0
        
        Q = 2*np.ones((Mesh.num_elements(self.mesh), len(self.quad_rule), 1))
        
        energy = field_operator.integrate(f, self.mesh.coords, self.mesh.blocks['block_0'], U, Q)
        print(f"{energy:.12e}")
        assert energy == pytest.approx(28.0994)

    def test_nonexistent_field_id_gets_error(self):
        spaces = PkField(self.coord_degree, self.mesh),
        integrand_signature = Gradient(1), # there is no field 1
        with pytest.raises(AssertionError):
            field_operator = FieldOperator(spaces, integrand_signature, self.mesh, self.quad_rule)

    def test_jit_and_grad(self):
        k = 2
        spaces = PkField(k, self.mesh),
        integrand_signature = Gradient(0),
        field_operator = FieldOperator(spaces, integrand_signature, self.mesh, self.quad_rule)

        def f(dudX):
            return 0.5*np.dot(dudX, dudX)

        @jax.jit
        def energy(U):
            return field_operator.integrate(f, self.mesh.coords, self.mesh.blocks['block_0'], U)

        target_grad = np.array([0.1, 0.01])
        V = spaces[0].coords@target_grad

        e = energy(V)
        assert e == pytest.approx(0.5*np.dot(target_grad, target_grad)*self.length*self.height)

        force = jax.jit(jax.grad(energy))
        F = force(V)
        assert sum(F) == pytest.approx(0.0)


class TestDGField:
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(1)
    mesh = Mesh.construct_structured_mesh(3, 3, [0.0, 2.0], [0.0, 1.0])

    def test_dg_interpolation(self):
        k = 1
        space = DG_PkField(k, self.mesh)

        # make a DG field that is u = 0.1*x on x < 1, u = 2 on x > 1
        def field_values(el_coords):
            centroid = np.mean(el_coords, axis=0)
            return np.where(centroid[0] < 1.0, 0.1*el_coords[:, 0], 2.0*np.ones_like(el_coords[:, 0]))

        U = jax.vmap(field_values)(space.coords)

        coord_space = PkField(1, self.mesh)
        field_operator = FieldOperator((space, coord_space), (Value(0), Value(1)), self.mesh, self.quad_rule)
        def f(u, x):
            return u, x

        # elements in left-hand side are [0:4] -> u = 0.1*x
        u_q, x_q = field_operator.evaluate(f, self.mesh.coords, np.arange(4), U, self.mesh.coords)
        Uq_expected = 0.1*x_q[..., 0]
        assert u_q == pytest.approx(Uq_expected)

        # elements in right-hand side [4:7] u = 2.0
        u_q, x_q = field_operator.evaluate(f, self.mesh.coords, np.arange(4, 8), U, self.mesh.coords)
        assert u_q == pytest.approx(2.0)

    def test_dg_gradient_interpolation(self):
        k = 1
        self.mesh = Mesh.construct_structured_mesh(3, 3, [0.0, 2.0], [0.0, 1.0])
        space = DG_PkField(k, self.mesh)

        # make a DG field that is u = 0.1*x on x < 1, u = 2 on x > 1
        def field_values(el_coords):
            centroid = np.mean(el_coords, axis=0)
            return np.where(centroid[0] < 1.0, 0.1*el_coords[:, 0], 2.0*np.ones_like(el_coords[:, 0]))

        U = jax.vmap(field_values)(space.coords)

        coord_space = PkField(1, self.mesh)
        field_operator = FieldOperator((space, coord_space), (Gradient(0), Value(1)), self.mesh, self.quad_rule)
        def f(dudX, x):
            return dudX, x

        # elements in left-hand side are [0:4] -> grad u = [0.1, 0.0]
        dudX_q, _ = field_operator.evaluate(f, self.mesh.coords, np.arange(4), U, self.mesh.coords)
        for dudX in dudX_q.reshape(-1, 2):
            assert dudX == pytest.approx(np.array([0.1, 0.0]))

        # elements in right-hand side [4:7] -> grad u = 0
        dudX_q, _ = field_operator.evaluate(f, self.mesh.coords, np.arange(4, 8), U, self.mesh.coords)
        assert dudX_q == pytest.approx(0.0)

def test_parameterized_elasticity():
    mesh = Mesh.construct_structured_mesh(5, 3, [0.0, 1.0], [0.0, 1.0])
    ne = Mesh.num_elements(mesh)
    blocks = {'all': np.arange(Mesh.num_elements(mesh)),
              'left': np.arange(ne//2),
              'right': np.arange(ne//2, ne)}
    quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(2)
    spaces = PkField(1, mesh), DG_PkField(0, mesh)
    integrand_signature = Gradient(0), Value(1)
    field_operator = FieldOperator(spaces, integrand_signature, mesh, quad_rule)
    
    lam = 3.0
    
    def f(dudX, mu):
        strain = 0.5*(dudX + dudX.T)
        return mu*np.tensordot(strain, strain) + 0.5*lam*np.trace(strain)**2
    
    target_disp_grad = np.array([[0.1, 0.01],
                                 [0.05, 0.3]])
    U = np.einsum('aj, ij', mesh.coords, target_disp_grad + np.identity(2)) - mesh.coords
    
    mu_left = 1.0
    mu_right = 2.0
    mu = np.zeros(spaces[1].field_shape)
    mu = mu.at[blocks['left']].set(mu_left)
    mu = mu.at[blocks['right']].set(mu_right)
    
    def energy(U, mu):
        return field_operator.integrate(f, mesh.coords, blocks['all'], U, mu)
    
    R = jax.grad(energy, 0)(U, mu)
    assert R[6] == pytest.approx(np.zeros(2))
    assert R[8] == pytest.approx(np.zeros(2))
    

    stresses = field_operator.evaluate(jax.grad(f, 0), mesh.coords, blocks['all'], U, mu)
    strain = 0.5*(target_disp_grad + target_disp_grad.T)
    stress_left_exact = 2.0*mu_left*strain + lam*np.trace(strain)*np.identity(2)
    stress_right_exact = 2.0*mu_right*strain + lam*np.trace(strain)*np.identity(2)
    for stress in stresses[blocks['left']].reshape(-1, 2, 2):
        assert stress == pytest.approx(stress_left_exact)
    for stress in stresses[blocks['right']].reshape(-1, 2, 2):
        assert stress == pytest.approx(stress_right_exact)
