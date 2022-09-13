import pathlib # to reach sample mesh file
import sys
import unittest

from optimism.JaxConfig import *
try:
    from optimism import ReadExodusMesh
except ImportError:
    print('netCDF4 package not installed')
    print('Support for Exodus mesh format reader is disabled')
    
from optimism import FunctionSpace
from optimism.material import LinearElastic as MaterialModel
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from optimism.test import TestFixture
from optimism import Mechanics

haveNetCDF = 'netCDF4' in sys.modules
skipMessage = 'netCDF4 not installed, exodus reader disabled'


TEST_FILE = pathlib.Path(__file__).parent.joinpath('patch_2_blocks.g')


class TestMeshReadData(TestFixture.TestFixture):
    # output from explore:
    
    # Number of coordinates per node       =           2
    # Number of nodes                      =          45
    # Number of elements                   =          16
    # Number of element blocks             =           2


    # Number of nodal point sets           =           4
    #    Length of node list               =          28
    #    Length of distribution list       =          28
    # Number of element side sets          =           2
    #    Length of element list            =           6
    #    Length of node list               =          18
    #    Length of distribution list       =          18  

    def setUp(self):
        self.mesh = ReadExodusMesh.read_exodus_mesh(TEST_FILE)

    @TestFixture.unittest.skipIf(not haveNetCDF, skipMessage)
    def test_entity_counts(self):
        readNodes = self.mesh.coords.shape[0]
        self.assertEqual(readNodes, 45)

        readElements = self.mesh.conns.shape[0]
        self.assertEqual(readElements, 16)

        readNodeSets = len(self.mesh.nodeSets)
        self.assertEqual(readNodeSets, 4)

        readSideSets = len(self.mesh.sideSets)
        self.assertEqual(readSideSets, 2)


    @TestFixture.unittest.skipIf(not haveNetCDF, skipMessage)
    def test_all_sets_named(self):
        for ns in self.mesh.nodeSets:
            self.assertGreater(len(ns), 0)

        for ss in self.mesh.sideSets:
            self.assertGreater(len(ss), 0)

            
    @TestFixture.unittest.skipIf(not haveNetCDF, skipMessage)
    def test_node_set_sizes(self):
        # EXPLORE>  list nsets
        
        #  Set           1 (#1):           5 nodes (index=1)      name = "left"
        #  Set           2 (#2):           9 nodes (index=6)      name = "bottom"
        #  Set           3 (#3):           5 nodes (index=15)     name = "right"
        #  Set           4 (#4):           9 nodes (index=20)     name = "top"
        self.assertEqual(self.mesh.nodeSets["left"].size, 5)
        self.assertEqual(self.mesh.nodeSets["bottom"].size, 9)
        self.assertEqual(self.mesh.nodeSets["right"].size, 5)
        self.assertEqual(self.mesh.nodeSets["top"].size, 9)


    @TestFixture.unittest.skipIf(not haveNetCDF, skipMessage)
    def test_side_set_sizes(self):
        # EXPLORE> list ssets

        #  Set           1 (#1):           2 elements (index=1)           6 nodes/df (index=1)  name = "right"
        #  Set           2 (#2):           4 elements (index=3)          12 nodes/df (index=7)  name = "top"
        self.assertEqual(self.mesh.sideSets["right"].shape[0], 2)
        self.assertEqual(self.mesh.sideSets["top"].shape[0], 4)



class TestMeshReadPatchTest(TestFixture.TestFixture):
    def setUp(self):
        self.mesh = ReadExodusMesh.read_exodus_mesh(TEST_FILE)
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)

        self.E = 1.0
        self.nu = 0.3
        props = {'elastic modulus': self.E,
                 'poisson ratio': self.nu}
        materialModel = MaterialModel.create_material_model_functions(props)

        mechBvp = Mechanics.create_mechanics_functions(self.fs,
                                                       "plane strain",
                                                       materialModel)

        self.compute_strain_energy = mechBvp.compute_strain_energy
        self.internalVariables = mechBvp.compute_initial_state()

        
    @TestFixture.unittest.skipIf(not haveNetCDF, skipMessage)        
    def test_dirichlet_patch_test(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='left', component=1),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1),
                FunctionSpace.EssentialBC(nodeSet='right', component=0),
                FunctionSpace.EssentialBC(nodeSet='right', component=1),
                FunctionSpace.EssentialBC(nodeSet='top', component=0),
                FunctionSpace.EssentialBC(nodeSet='top', component=1)]

        targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        U = self.mesh.coords@targetDispGrad.T
        
        dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)
            
        # Uu is U_unconstrained
        Ubc = dofManager.get_bc_values(U)
        @jit
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            return self.compute_strain_energy(U, self.internalVariables)
        
        grad_func = jit(grad(objective))
        
        Uu = newton_solve(objective, dofManager.get_unknown_values(U))
        U = dofManager.create_field(Uu, Ubc)
           
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,2,2):
            self.assertArrayNear(dg, targetDispGrad, 14)

        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)


if __name__ == '__main__':
    unittest.main()
