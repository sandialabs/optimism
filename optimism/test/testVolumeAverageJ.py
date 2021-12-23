import numpy as onp

from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mechanics
from optimism import Mesh
from optimism.test import MeshFixture
from optimism import QuadratureRule


class TestVolumeAverage(MeshFixture.MeshFixture):

    def setUp(self):
        self.Nx = 2
        self.Ny = 3
        xRange = [0.,2.]
        yRange = [0.,1.]
        
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x : self.targetDispGrad.dot(x))
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2)
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=4)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)

        self.UWithConstantJ = self.mesh.coords@self.targetDispGrad.T
        self.UWithLinearJ = np.column_stack( (0.1*self.mesh.coords[:,0]**2 + 0.3*self.mesh.coords[:,1]**2,
                                              self.mesh.coords[:,0]) )
        self.UWithQuadraticJ = np.column_stack((0.1*self.mesh.coords[:,0]**2, 0.3*self.mesh.coords[:,1]**2))

        masterJConstant = Interpolants.make_master_tri_element(degree=0)
        masterJLinear = Interpolants.make_master_tri_element(degree=1)

        self.shapesJConstant = Interpolants.compute_shapes_on_tri(masterJConstant, quadRule.xigauss)
        self.shapesJLinear = Interpolants.compute_shapes_on_tri(masterJLinear, quadRule.xigauss)

        
    def test_constant_J_projection_exact_for_constant_J_field(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.UWithConstantJ)
        defGrads = dispGrads + np.identity(2)
        Js = np.linalg.det(defGrads)
        
        projectedDispGrads = vmap(Mechanics.volume_average_J_gradient_transformation, (0,0,None))(dispGrads, self.fs.vols, self.shapesJConstant)
        projectedDefGrads = projectedDispGrads + np.identity(2)
        JBars = np.linalg.det(projectedDefGrads)

        #print('dispGrads=',dispGrads)
        #print('J=',Js)
        #print('JBar=',JBars)
        
        self.assertArrayNear(Js, JBars, 13)


    def test_linear_J_projection_exact_for_constant_J_field(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.UWithConstantJ)
        defGrads = dispGrads + np.identity(2)
        Js = np.linalg.det(defGrads)
        
        projectedDispGrads = vmap(Mechanics.volume_average_J_gradient_transformation, (0,0,None))(dispGrads, self.fs.vols, self.shapesJLinear)
        projectedDefGrads = projectedDispGrads + np.identity(2)
        JBars = np.linalg.det(projectedDefGrads)
        
        self.assertArrayNear(Js, JBars, 13)


    def test_constant_J_projection_inexact_for_linear_J_field(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.UWithLinearJ)
        defGrads = dispGrads + np.identity(2)
        Js = np.linalg.det(defGrads)
        
        projectedDispGrads = vmap(Mechanics.volume_average_J_gradient_transformation, (0,0,None))(dispGrads, self.fs.vols, self.shapesJConstant)
        projectedDefGrads = projectedDispGrads + np.identity(2)
        JBars = np.linalg.det(projectedDefGrads)

        with self.assertRaises(AssertionError):
            self.assertArrayNear(Js, JBars, 13)


    def test_linear_J_projection_exact_for_linear_J_field(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.UWithLinearJ)
        defGrads = dispGrads + np.identity(2)
        Js = np.linalg.det(defGrads)
        
        projectedDispGrads = vmap(Mechanics.volume_average_J_gradient_transformation, (0,0,None))(dispGrads, self.fs.vols, self.shapesJLinear)
        projectedDefGrads = projectedDispGrads + np.identity(2)
        JBars = np.linalg.det(projectedDefGrads)
        
        self.assertArrayNear(Js, JBars, 13)
        

    def test_linear_J_projection_inexact_for_quadratic_J_field(self):
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, self.UWithQuadraticJ)
        defGrads = dispGrads + np.identity(2)
        Js = np.linalg.det(defGrads)
        
        projectedDispGrads = vmap(Mechanics.volume_average_J_gradient_transformation, (0,0,None))(dispGrads, self.fs.vols, self.shapesJLinear)
        projectedDefGrads = projectedDispGrads + np.identity(2)
        JBars = np.linalg.det(projectedDefGrads)

        with self.assertRaises(AssertionError):
            self.assertArrayNear(Js, JBars, 13)

        

if __name__ == '__main__':
    MeshFixture.unittest.main()
