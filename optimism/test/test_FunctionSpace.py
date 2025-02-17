import unittest

import jax
import jax.numpy as np

from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mesh
from optimism import QuadratureRule
from . import TestFixture
from . import MeshFixture


class TestFunctionSpaceFixture(TestFixture.TestFixture):

    def setUp(self):
        self.coords = np.array([[-0.6162298 ,  4.4201174],
                                [-2.2743905 ,  4.53892   ],
                                [ 2.0868123 ,  0.68486094]])
        self.conn = np.arange(0, 3)
        self.parentElement = Interpolants.make_parent_element_2d(degree=1)


    def test_mass_matrix_exactly_integrated(self):
        UNodal = np.ones(3) # value of U doesn't matter
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        stateVar = None
        shape, shapeGrad = Interpolants.compute_shapes(self.parentElement, quadRule.xigauss)
        vol = FunctionSpace.compute_element_volumes(self.coords,
                                                    self.conn,
                                                    self.parentElement,
                                                    shape,
                                                    quadRule.wgauss)
        def f(u, gradu, state, X): return 0.5*u*u
        compute_mass = jax.hessian(lambda U: FunctionSpace.integrate_element(U, self.coords, stateVar, shape, shapeGrad, vol, self.conn, f, modify_element_gradient=FunctionSpace.default_modify_element_gradient))
        M = compute_mass(UNodal)
        area = 0.5*np.cross(self.coords[1,:] - self.coords[0,:],
                            self.coords[2,:] - self.coords[0,:])
        MExact = area/12.0*np.array([[2., 1., 1.],
                                     [1., 2., 1.],
                                     [1., 1., 2.]])
        self.assertTrue(np.allclose(M, MExact, atol=1e-14, rtol=1e-10))


    def test_mass_matrix_inexactly_integrated_with_low_order_quadrature(self):
        UNodal = np.ones(3) # value of U doesn't matter
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        stateVar = None
        shape, shapeGrad = Interpolants.compute_shapes(self.parentElement, quadRule.xigauss)
        vol = FunctionSpace.compute_element_volumes(self.coords,
                                                    self.conn,
                                                    self.parentElement,
                                                    shape,
                                                    quadRule.wgauss)
        def f(u, gradu, state, X): return 0.5*u*u
        compute_mass = jax.hessian(lambda U: FunctionSpace.integrate_element(U, self.coords, stateVar, shape, shapeGrad, vol, self.conn, f, modify_element_gradient=FunctionSpace.default_modify_element_gradient))
        M = compute_mass(UNodal)
        area = 0.5*np.cross(self.coords[1,:] - self.coords[0,:],
                            self.coords[2,:] - self.coords[0,:])
        MExact = area/12.0*np.array([[2., 1., 1.],
                                     [1., 2., 1.],
                                     [1., 1., 2.]])
        self.assertFalse(np.allclose(M, MExact, atol=1e-14, rtol=1e-10))

    
class TestFunctionSpaceSingleQuadPointFixture(MeshFixture.MeshFixture):

    def setUp(self):
        self.quadratureRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)

        # mesh
        self.Nx = 7
        self.Ny = 7
        self.xRange = [0.,1.]
        self.yRange = [0.,1.]
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])        
        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, self.xRange, self.yRange,
                                                      lambda x : self.targetDispGrad.dot(x))

        # function space
        self.fs = FunctionSpace.construct_function_space(self.mesh,
                                                         self.quadratureRule)

        nElements = Mesh.num_elements(self.mesh)
        nQuadPoints = len(self.quadratureRule)
        self.state = np.zeros((nElements,nQuadPoints,1))
        self.dt = 0.0


    def test_element_volume_single_point_quadrature(self):
        elementVols = np.sum(self.fs.vols, axis=1)
        nElements = Mesh.num_elements(self.mesh)
        self.assertArrayNear(elementVols, np.ones(nElements)*0.5/((self.Nx-1)*(self.Ny-1)), 14)


    def test_linear_reproducing_single_point_quadrature(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.U)
        nElements = Mesh.num_elements(self.mesh)
        npts = self.quadratureRule.xigauss.shape[0]
        exact = np.tile(self.targetDispGrad, (nElements, npts, 1, 1))
        self.assertTrue(np.allclose(dispGrads, exact))


    def test_integrate_constant_field_single_point_quadrature(self):
        integralOfOne = FunctionSpace.integrate_over_block(self.fs,
                                                           self.U,
                                                           self.state,
                                                           self.dt,
                                                           lambda u, gradu, state, X, dt: 1.0,
                                                           self.mesh.blocks['block'])
        self.assertNear(integralOfOne, 1.0, 14)


    def test_integrate_linear_field_single_point_quadrature(self):
        
        Ix = FunctionSpace.integrate_over_block(self.fs,
                                                self.U,
                                                self.state,
                                                self.dt,
                                                lambda u, gradu, state, X, dt: gradu[0,0],
                                                self.mesh.blocks['block'])
        # displacement at x=1 should match integral
        idx = np.argmax(self.mesh.coords[:,0])
        expected = self.U[idx,0]*(self.yRange[1] - self.yRange[0])
        self.assertNear(Ix, expected, 14)
        
        Iy = FunctionSpace.integrate_over_block(self.fs,
                                                self.U,
                                                self.state,
                                                self.dt,
                                                lambda u, gradu, state, X, dt: gradu[1,1],
                                                self.mesh.blocks['block'])
        idx = np.argmax(self.mesh.coords[:,1])
        expected = self.U[idx,1]*(self.xRange[1] - self.xRange[0])
        self.assertNear(Iy, expected, 14)

        
class TestFunctionSpaceMultiQuadPointFixture(MeshFixture.MeshFixture):

    def setUp(self):
        self.quadratureRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)

        # mesh
        self.Nx = 7
        self.Ny = 7
        self.xRange = [0.,1.]
        self.yRange = [0.,1.]
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])        
        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, self.xRange, self.yRange,
                                                      lambda x : self.targetDispGrad.dot(x))
        # function space
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadratureRule)
        nElements = Mesh.num_elements(self.mesh)
        nQuadPoints = len(self.quadratureRule)
        self.state = np.zeros((nElements,nQuadPoints,))
        self.dt = 0.0


    def test_element_volume_multi_point_quadrature(self):
        elementVols = np.sum(self.fs.vols, axis=1)
        nElements = Mesh.num_elements(self.mesh)
        self.assertArrayNear(elementVols, np.ones(nElements)*0.5/((self.Nx-1)*(self.Ny-1)), 14)


    def test_linear_reproducing_multi_point_quadrature(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.U)
        nElements = Mesh.num_elements(self.mesh)
        npts = self.quadratureRule.xigauss.shape[0]
        exact = np.tile(self.targetDispGrad, (nElements, npts, 1, 1))
        self.assertTrue(np.allclose(dispGrads, exact))


    def test_integrate_constant_field_multi_point_quadrature(self):
        integralOfOne = FunctionSpace.integrate_over_block(self.fs,
                                                           self.U,
                                                           self.state,
                                                           self.dt,
                                                           lambda u, gradu, state, X, dt: 1.0,
                                                           self.mesh.blocks['block'])
        self.assertNear(integralOfOne, 1.0, 14)


    def test_integrate_linear_field_multi_point_quadrature(self):
        Ix = FunctionSpace.integrate_over_block(self.fs,
                                                self.U,
                                                self.state,
                                                self.dt,
                                                lambda u, gradu, state, X, dt: gradu[0,0],
                                                self.mesh.blocks['block'])
        idx = np.argmax(self.mesh.coords[:,0])
        expected = self.U[idx,0]*(self.yRange[1] - self.yRange[0])
        self.assertNear(Ix, expected, 14)
        
        Iy = FunctionSpace.integrate_over_block(self.fs,
                                                self.U,
                                                self.state,
                                                self.dt,
                                                lambda u, gradu, state, X, dt: gradu[1,1],
                                                self.mesh.blocks['block'])
        idx = np.argmax(self.mesh.coords[:,1])
        expected = self.U[idx,1]*(self.xRange[1] - self.xRange[0])
        self.assertNear(Iy, expected, 14)


    def test_integrate_over_half_block(self):
        nElements = Mesh.num_elements(self.mesh)
        # this test will only work with an even number of elements
        # put this in so that if test is modified to odd number,
        # we understand why it fails
        self.assertEqual(nElements % 2, 0)
        
        blockWithHalfTheVolume = slice(0,nElements//2)
        integral = FunctionSpace.integrate_over_block(self.fs,
                                                      self.U,
                                                      self.state,
                                                      self.dt,
                                                      lambda u, gradu, state, X, dt: 1.0,
                                                      blockWithHalfTheVolume)
        self.assertNear(integral, 1.0/2.0, 14)


    def test_integrate_over_half_block_indices(self):
        nElements = Mesh.num_elements(self.mesh)
        # this test will only work with an even number of elements
        # put this in so that if test is modified to odd number,
        # we understand why it fails
        self.assertEqual(nElements % 2, 0)
        
        blockWithHalfTheVolume = np.arange(nElements//2)
        
        integral = FunctionSpace.integrate_over_block(self.fs,
                                                      self.U,
                                                      self.state,
                                                      self.dt,
                                                      lambda u, gradu, state, X, dt: 1.0,
                                                      blockWithHalfTheVolume)
        self.assertNear(integral, 1.0/2.0, 14)
        
        
    def test_jit_on_integration(self):
        integrate_jit = jax.jit(FunctionSpace.integrate_over_block, static_argnums=(4,))
        I = integrate_jit(self.fs, self.U, self.state, self.dt, lambda u, gradu, state, X, dt: 1.0, self.mesh.blocks['block'])
        self.assertNear(I, 1.0, 14)

        
    def test_jit_and_jacrev_on_integration(self):
        F = jax.jit(jax.jacrev(FunctionSpace.integrate_over_block, 1), static_argnums=(4,))
        dI = F(self.fs, self.U, self.state, self.dt, lambda u, gradu, state, X, dt: 0.5*np.tensordot(gradu, gradu),
               self.mesh.blocks['block'])
        nNodes = self.mesh.coords.shape[0]
        interiorNodeIds = np.setdiff1d(np.arange(nNodes), self.mesh.nodeSets['all_boundary'])
        self.assertArrayNear(dI[interiorNodeIds,:], np.zeros_like(self.U[interiorNodeIds,:]), 14)
                
        
class ParameterizationTestSuite(MeshFixture.MeshFixture):
    def setUp(self) -> None:
        self.quadratureRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)

        # mesh
        self.Nx = 7
        self.Ny = 7
        self.xRange = [0.,3.]
        self.yRange = [0.,1.]
        self.A = np.array([[0.1, -0.2],
                           [0.4, -0.1]])
        self.b = 2.5
        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, self.xRange, self.yRange,
                                                      lambda x : np.array([0.0, x[0]]))
        # function space
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadratureRule)
        nElements = Mesh.num_elements(self.mesh)
        nQuadPoints = len(self.quadratureRule)
        self.state = np.zeros((nElements,nQuadPoints,))
        self.dt = 0.0

    def test_integrate_with_parameter(self):
        def centroid(v):
            return np.average(v, axis=0)

        xc = jax.vmap(lambda conn, coords: centroid(coords[conn, :]), (0, None))(self.mesh.conns, self.mesh.coords)
        weights = np.ones(Mesh.num_elements(self.mesh), dtype=np.float64)
        weights = weights.at[xc[:,0] < self.xRange[1]/2].set(2.0)
        f = FunctionSpace.integrate_over_block(self.fs, self.U, self.state, self.dt, lambda u, dudx, q, x, dt, p: p, self.mesh.blocks['block'], weights)
        exact = 1.5*self.xRange[1]*self.yRange[1]
        self.assertAlmostEqual(f, exact, 12)


class TestFunctionSpaceWithQuadraticLagrangeElementFixture(MeshFixture.MeshFixture):
    def setUp(self):
        self.quadratureRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)

        # mesh
        self.Nx = 2
        self.Ny = 2
        self.xRange = [0.,1.]
        self.yRange = [0.,1.]
        self.targetDispHessian = np.array([[0.1, -0.2, 0.6],[-0.4, 0.5, -0.6]])

        # u_i = 0.5 G_ijk x_j x_k
        initial_disp_func = lambda x : 0.5*self.targetDispHessian.dot(np.array([x[0]*x[0], x[1]*x[1], 2.0*x[0]*x[1]]))
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, self.xRange, self.yRange,
                                                 initial_disp_func)
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, 
                                                                    interpolationType=Interpolants.InterpolationType.LAGRANGE)
        self.U = jax.vmap(initial_disp_func)(self.mesh.coords)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadratureRule)

    def test_quadratic_reproducing_gradient_and_hessian(self):
        # du_i / dx_p = G_ipk x_k
        def compute_expected_displacement_grads(X, elemConnectivity, elemShapeFunctions):
            elemNodalX = X[elemConnectivity]
            def compute_qp_displacement_grad(NodalX, shapeFuncs):
                x = np.dot(shapeFuncs, NodalX)
                return np.array([[self.targetDispHessian[0,0]*x[0] + self.targetDispHessian[0,2]*x[1], self.targetDispHessian[0,2]*x[0] + self.targetDispHessian[0,1]*x[1]],
                                 [self.targetDispHessian[1,0]*x[0] + self.targetDispHessian[1,2]*x[1], self.targetDispHessian[1,2]*x[0] + self.targetDispHessian[1,1]*x[1]]])
            return jax.vmap(compute_qp_displacement_grad, (None, 0))(elemNodalX, elemShapeFunctions)

        from optimism import Mechanics
        dispGradsAndHessians = FunctionSpace.compute_field_gradient(self.fs, self.U, modify_element_gradient=Mechanics.plane_strain_gradient_transformation)

        # check displacement gradients
        dispGrads = dispGradsAndHessians[:,:,0:2,0:2]
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quadratureRule.xigauss)
        exact = jax.vmap(compute_expected_displacement_grads, (None, 0, None))(self.mesh.coords, self.mesh.conns, shapeOnRef.values)
        self.assertTrue(np.allclose(dispGrads, exact))

        # check displacement hessians
        dispHessians = dispGradsAndHessians[:,:,0:2,3:]
        nElements = Mesh.num_elements(self.mesh)
        npts = self.quadratureRule.xigauss.shape[0]
        exact = np.tile(self.targetDispHessian, (nElements, npts, 1, 1))
        self.assertTrue(np.allclose(dispHessians, exact))



if __name__ == '__main__':
    unittest.main()
