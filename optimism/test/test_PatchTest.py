import unittest

import jax
import jax.numpy as np

from optimism import FunctionSpace
from optimism import Interpolants
from optimism.material import LinearElastic as MatModel
from optimism import Mesh
from optimism import Mechanics
from optimism.Timer import Timer
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from . import MeshFixture

kappa = 1.0
mu = 0.25

E = 9 * kappa * mu / (3 * kappa + mu)
nu = 0.5 * (3 * kappa - 2 * mu) / (3 * kappa + mu)

props = {'elastic modulus': E,
         'poisson ratio': nu,
         'strain measure': 'linear'}

targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 

class PatchTestFixture(MeshFixture.MeshFixture):
    def initializeLinearPatchTestFromMesh(self, mesh, quadratureDegree):
        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=quadratureDegree)
        self.fs = FunctionSpace.construct_function_space(mesh, self.quadRule)

        materialModel = MatModel.create_material_model_functions(props)

        mcxFuncs = \
            Mechanics.create_mechanics_functions(self.fs,
                                                 "plane strain",
                                                 materialModel)

        self.compute_energy = jax.jit(mcxFuncs.compute_strain_energy)
        self.internals = mcxFuncs.compute_initial_state()

    def initializeQuadraticPatchTestFromMesh(self, mesh):
        alpha = 2.0
        beta = 1.0

        self.UTarget = jax.vmap( lambda x : targetDispGrad@x + 
                                 np.array([alpha * x[0]*x[0], beta * x[1]*x[1]]) )(mesh.coords)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(mesh, self.quadRule)

        materialModel = MatModel.create_material_model_functions(props)

        self.b = -(2*kappa + 8*mu/3.0) * np.array([alpha, beta])

        mcxFuncs = Mechanics.create_mechanics_functions(self.fs, "plane strain", materialModel)

        self.compute_energy = jax.jit(mcxFuncs.compute_strain_energy)
        self.internals = mcxFuncs.compute_initial_state()

    def assertLinearPatchTestPasses(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.UTarget.shape[1], ebcs)
        Ubc = dofManager.get_bc_values(self.UTarget)
        
        @jax.jit
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            return self.compute_energy(U, self.internals)
        
        with Timer(name="NewtonSolve"):
            Uu, solverSuccess = newton_solve(objective, dofManager.get_unknown_values(self.UTarget))
            self.assertTrue(solverSuccess)

        U = dofManager.create_field(Uu, Ubc)
            
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,2,2):
            self.assertArrayNear(dg, targetDispGrad, 14)

        grad_func = jax.jit(jax.grad(objective))
        Uu = dofManager.get_unknown_values(U)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)

    def assertQuadraticPatchTestPasses(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        self.dofManager = FunctionSpace.DofManager(self.fs, self.UTarget.shape[1], ebcs)
        Ubc = self.dofManager.get_bc_values(self.UTarget)
        
        def constant_body_force_potential(U, internals, b):
            dtUnused = 0.0
            f = lambda u, du, q, x, dt: -np.dot(b, u)
            return FunctionSpace.integrate_over_block(self.fs, U, internals, dtUnused, f, slice(None))

        def objective(Uu):
            U = self.dofManager.create_field(Uu, Ubc)
            return self.compute_energy(U, self.internals) + constant_body_force_potential(U, self.internals, self.b)
        
        with Timer(name="NewtonSolve"):
            Uu, solverSuccess = newton_solve(objective, 0.0*self.dofManager.get_unknown_values(self.UTarget))
            self.assertTrue(solverSuccess)

        U = self.dofManager.create_field(Uu, Ubc)

        self.assertArrayNear(U, self.UTarget, 13)

        grad_func = jax.jit(jax.grad(objective))
        Uu = self.dofManager.get_unknown_values(U)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)


class LinearPatchTestLinearElements(PatchTestFixture):
    
    def setUp(self):
        self.Nx = 7
        self.Ny = 7
        xRange = [0.,1.]
        yRange = [0.,1.]
        
        self.mesh, self.UTarget = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x : targetDispGrad.dot(x))
        self.initializeLinearPatchTestFromMesh(self.mesh, quadratureDegree=1)

        
    def test_dirichlet_patch_test(self):
        self.assertLinearPatchTestPasses()


    def test_neumann_patch_test(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        self.UTarget = np.zeros(self.UTarget.shape)
        dofManager = FunctionSpace.DofManager(self.fs, self.UTarget.shape[1], ebcs)
        Ubc = dofManager.get_bc_values(self.UTarget)
        
        sigma = np.array([[1.0, 0.0], [0.0, 0.0]])
        traction_func = lambda x, n: np.dot(sigma, n)     
        quadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        
        @jax.jit
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            internalPotential = self.compute_energy(U, self.internals)
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, quadRule, self.mesh.sideSets['right'], traction_func)
            loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, quadRule, self.mesh.sideSets['top'], traction_func)
            return internalPotential + loadPotential
        
        with Timer(name="NewtonSolve"):
            Uu, solverSuccess = newton_solve(objective, dofManager.get_unknown_values(self.UTarget))
            self.assertTrue(solverSuccess)

        self.UTarget = dofManager.create_field(Uu, Ubc)

        # exact solution
        modulus1 = (1.0 - nu**2)/E
        modulus2 = -nu*(1.0+nu)/E
        UExact = np.column_stack( ((modulus1*sigma[0, 0] + modulus2*sigma[1, 1])*self.mesh.coords[:,0],
                                   (modulus2*sigma[0, 0] + modulus1*sigma[1, 1])*self.mesh.coords[:,1]) )
        
        self.assertArrayNear(self.UTarget, UExact, 14)



class LinearPatchTestQuadraticElements(PatchTestFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]
        
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                            lambda x : targetDispGrad.dot(x))
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, createNodeSetsFromSideSets=True)
        self.initializeLinearPatchTestFromMesh(self.mesh, quadratureDegree=2)
        self.UTarget = self.mesh.coords@targetDispGrad.T
        
    
    def test_dirichlet_patch_test_with_quadratic_elements(self):
        self.assertLinearPatchTestPasses()


    def test_dirichlet_patch_test_with_quadratic_elements_and_constant_jac_projection(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.UTarget.shape[1], ebcs)
        Ubc = dofManager.get_bc_values(self.UTarget)

        elementForJ = Interpolants.make_parent_element_2d(degree=0)
        shapesForJ, _ = Interpolants.shape2d(elementForJ.degree, elementForJ.coordinates, self.quadRule.xigauss)
        
        def modify_grad(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
            elemGrads = Mechanics.volume_average_J_gradient_transformation(elemGrads, elemVols, shapesForJ)
            return Mechanics.plane_strain_gradient_transformation(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords)

        @jax.jit
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            return self.compute_energy(U, self.internals)
        
        with Timer(name="NewtonSolve"):
            Uu, solverSuccess = newton_solve(objective, dofManager.get_unknown_values(self.UTarget))
            self.assertTrue(solverSuccess)

        U = dofManager.create_field(Uu, Ubc)
            
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U, modify_grad)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,*dispGrads.shape[2:]):
            self.assertArrayNear(dg[:2,:2], targetDispGrad, 14)

        grad_func = jax.jit(jax.grad(objective))
        Uu = dofManager.get_unknown_values(U)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)


class QuadraticPatchTestQuadraticElements(PatchTestFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]
        
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                            lambda x : targetDispGrad.dot(x))

        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, createNodeSetsFromSideSets=True)
        self.initializeQuadraticPatchTestFromMesh(self.mesh)

    def test_dirichlet_patch_test_with_quadratic_elements(self):
        self.assertQuadraticPatchTestPasses()


class LinearPatchTestQuadraticLagrangeElements(PatchTestFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]
        
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                            lambda x : targetDispGrad.dot(x))
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, 
                                                                    interpolationType=Interpolants.InterpolationType.LAGRANGE,
                                                                    createNodeSetsFromSideSets=True)
        self.initializeLinearPatchTestFromMesh(self.mesh, quadratureDegree=2)
        self.UTarget = self.mesh.coords@targetDispGrad.T
    
    def test_dirichlet_patch_test_with_quadratic_elements(self):
        self.assertLinearPatchTestPasses()


class QuadraticPatchTestQuadraticLagrangeElements(PatchTestFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]
        
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                            lambda x : targetDispGrad.dot(x))

        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, 
                                                                    interpolationType=Interpolants.InterpolationType.LAGRANGE,
                                                                    createNodeSetsFromSideSets=True)
        self.initializeQuadraticPatchTestFromMesh(self.mesh)

    def test_dirichlet_patch_test_with_quadratic_elements(self):
        self.assertQuadraticPatchTestPasses()
        
        
if __name__ == '__main__':
    unittest.main()
