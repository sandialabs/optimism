from optimism.JaxConfig import *
from optimism.test import MeshFixture
from optimism import Mesh
from optimism import QuadratureRule
from optimism import Surface
from optimism import Timer


class TestSingleMeshFixture(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 4
        self.Ny = 4
        self.L = 1.2
        self.W = 1.5
        xRange = [0.,self.L]
        yRange = [0.,self.W]

        self.targetDispGrad = np.zeros((2,2))

        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x : self.targetDispGrad.dot(x))


        self.quadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        
    def test_integrate_perimeter(self):
        p = Surface.integrate_function_on_surface(self.quadRule,
                                                  self.mesh.sideSets['all_boundary'],
                                                  self.mesh,
                                                  lambda x, n: 1.0)
        self.assertNear(p, 2*(self.L+self.W), 14)

        
    def test_integrate_quadratic_fn_on_surface(self):
        I = Surface.integrate_function_on_surface(self.quadRule,
                                                  self.mesh.sideSets['top'],
                                                  self.mesh,
                                                  lambda x, n: x[0]**2)
        self.assertNear(I, self.L**3/3.0, 14)

        
    def test_integrate_function_on_surface_that_uses_coords_and_normal(self):
        I = Surface.integrate_function_on_surface(self.quadRule,
                                                  self.mesh.sideSets['all_boundary'],
                                                  self.mesh,
                                                  lambda x, n: np.dot(x,n))
        self.assertNear(I, 2*self.L*self.W, 14)

        
    def disable_test_edge_conn(self):
        print('coords=\n',self.mesh.coords)
        print('conns=\n', self.mesh.conns)
        edgeConn = Surface.create_edges_connectivity(self.mesh.coords, self.mesh.conns)
        print('edgeConn=\n',edgeConn)
        
if __name__ == '__main__':
    MeshFixture.unittest.main()

