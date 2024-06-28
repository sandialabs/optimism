import jax.numpy as np
import unittest
from unittest import mock

from optimism import FunctionSpace
from optimism import QuadratureRule
from . import MeshFixture
from optimism import Mechanics
from optimism import Mesh
from optimism.material import J2Plastic, Neohookean

class MechanicsFunctionsFixture(MeshFixture.MeshFixture):

    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.0, 1.0]
        yRange = [0.0, 1.0]

        self.targetDispGrad = np.array([[0.1, -0.2], [0.4, -0.1]])

        mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                 lambda x: self.targetDispGrad.dot(x))
        blocks = {'block0': np.array([0, 1, 2, 3]),
                  'block1': np.array([4, 5, 6, 7])}
        self.mesh = Mesh.mesh_with_blocks(mesh, blocks)
        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        props = {'elastic modulus': 1,
                 'poisson ratio': 0.25,
                 'yield strength': 0.1,
                 'hardening model': 'linear',
                 'hardening modulus': 0.1}

        materialModel0 = Neohookean.create_material_model_functions(props)
        materialModel1 = J2Plastic.create_material_model_functions(props)
        self.blockModels = {'block0': materialModel0, 'block1': materialModel1}
        self.mech_funcs = Mechanics.MechanicsFunctionsMultiBlock(self.fs, self.blockModels, Mechanics.plane_strain_gradient_transformation)


    def test_internal_variables_initialization_on_multi_block(self):
        nQuadPoints = QuadratureRule.len(self.quadRule)
        internals = self.mech_funcs.compute_initial_state()
        self.assertEqual(internals.shape, (self.mesh.num_elements, nQuadPoints, 10))
        self.assertArrayEqual(internals[0, 0], np.zeros(J2Plastic.NUM_STATE_VARS))
        self.assertArrayEqual(internals[4, 0], np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))

    def test_internal_variables_update_on_multi_block(self):
        internals = self.mech_funcs.compute_initial_state()
        dt = 1.0
        internalsNew = self.mech_funcs.compute_updated_internal_variables(self.U, internals, dt)
        self.assertEqual(internals.shape, internalsNew.shape)
        self.assertGreater(internalsNew[4,0,J2Plastic.EQPS], 0.05)

if __name__ == "__main__":
    unittest.main()
