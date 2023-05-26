import unittest

import jax
import jax.numpy as np

from optimism import FunctionSpace
from optimism.material import LinearElastic as MatModel
from optimism import Mesh
from optimism import Mechanics
from optimism.Timer import Timer
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from optimism.test import MeshFixture

E = 1.0
nu = 0.3
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'strain measure': 'green lagrange'}

class TractionPatch(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0., 1.]
        yRange = [0., 1.]
        
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        
        mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                            lambda x : self.targetDispGrad.dot(x))
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, createNodeSetsFromSideSets=True)

        self.UTarget = self.mesh.coords@self.targetDispGrad.T
        
        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        materialModel = MatModel.create_material_model_functions(props)

        mcxFuncs = Mechanics.create_mechanics_functions(
            self.fs, "plane strain", materialModel)

        self.compute_energy = jax.jit(mcxFuncs.compute_strain_energy)
        self.internals = mcxFuncs.compute_initial_state()
        
        dummyInternals = materialModel.compute_initial_state()
        dispGrad3D = np.zeros((3,3)).at[0:2,0:2].set(self.targetDispGrad)
        self.targetStress = jax.grad(materialModel.compute_energy_density)(dispGrad3D, dummyInternals, dt=0.0)

    
    def test_neumann_patch_test_with_quadratic_elements(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='left', component=1),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.UTarget.shape[1], ebcs)
        Ubc = dofManager.get_bc_values(self.UTarget)
        
        traction_func = lambda x, n: np.dot(self.targetStress[0:2, 0:2], n)
        
        
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            internalPotential = self.compute_energy(U, self.internals)
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, self.edgeQuadRule, self.mesh.sideSets['right'], traction_func)
            loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, self.edgeQuadRule, self.mesh.sideSets['top'], traction_func)
            return internalPotential + loadPotential
        
        with Timer(name="NewtonSolve"):
            Uu = newton_solve(objective, dofManager.get_unknown_values(self.UTarget))

        U = dofManager.create_field(Uu, Ubc)
            
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,2,2):
            self.assertArrayNear(dg, self.targetDispGrad, 14)

        grad_func = jax.jit(jax.grad(objective))
        Uu = dofManager.get_unknown_values(U)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)

if __name__ == "__main__":
    unittest.main()