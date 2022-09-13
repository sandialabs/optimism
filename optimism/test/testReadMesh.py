import pathlib

from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism.material import LinearElastic as MaterialModel
from optimism import Mesh
from optimism.Timer import Timer
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from optimism import Surface
from optimism.test import TestFixture
from optimism import ReadMesh
from optimism import Mechanics

TEST_FILE = pathlib.Path(__file__).parent.joinpath('patch.json')


class TestMeshReadData(TestFixture.TestFixture):
    # output from explore patch.g:
    
    # Number of nodes                      =          36
    # Number of elements                   =          50
    # Number of element blocks             =           1
    
    # Number of nodal point sets           =           5
    #    Length of node list               =          30
    #    Length of distribution list       =          30
    # Number of element side sets          =           5
    #    Length of element list            =          25
    #    Length of node list               =          50
    #    Length of distribution list       =          50
    

    def setUp(self):
        self.mesh = ReadMesh.read_json_mesh(TEST_FILE)

        
    def test_entity_counts(self):
        readNodes = self.mesh.coords.shape[0]
        self.assertEqual(readNodes, 36)

        readElements = self.mesh.conns.shape[0]
        self.assertEqual(readElements, 50)

        readNodeSets = len(self.mesh.nodeSets)
        self.assertEqual(readNodeSets, 5)

        readSideSets = len(self.mesh.sideSets)
        self.assertEqual(readSideSets, 5)


    def test_all_sets_named(self):
        for ns in self.mesh.nodeSets:
            self.assertGreater(len(ns), 0)

        for ss in self.mesh.sideSets:
            self.assertGreater(len(ss), 0)
    


def interpolate_nodal_field_on_edge(mesh, U, quadRule, edge):
    fieldIndex = Surface.get_field_index(edge, mesh.conns)  
    nodalValues = Surface.eval_field(U, fieldIndex)    
    return QuadratureRule.eval_at_iso_points(quadRule.xigauss, nodalValues)


def compute_traction_potential_energy_on_edge(mesh, U, quadRule, edge, load):
    uq = interpolate_nodal_field_on_edge(mesh, U, quadRule, edge)
    Xq = interpolate_nodal_field_on_edge(mesh, mesh.coords, quadRule, edge)
    tq = vmap(load)(Xq)
    edgeCoords = Surface.get_coords(mesh, edge)
    integrand = vmap(lambda u,t: u@t)(uq, tq)
    return -Surface.integrate_values(quadRule, edgeCoords, integrand)


def compute_traction_potential_energy(mesh, U, quadRule, edges, load):
    return np.sum( vmap(compute_traction_potential_energy_on_edge, (None,None,None,0,None))(mesh, U, quadRule, edges, load) )


class TestMeshReadPatchTest(TestFixture.TestFixture):
    def setUp(self):
        self.mesh = ReadMesh.read_json_mesh(TEST_FILE)
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
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
        fieldDim = U.shape[1]
        
        dofManager = FunctionSpace.DofManager(self.fs, fieldDim, ebcs)
            
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


    def test_neumann_patch_test(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        U = np.zeros(self.mesh.coords.shape)
        dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)
        Ubc = dofManager.get_bc_values(U)
        
        sigma11 = 1.0
        sigma22 = 0.0
        right_traction_func = lambda X: np.array([sigma11, 0.0])
        top_traction_func = lambda X: np.array([0.0, sigma22])       
        quadRule = QuadratureRule.create_quadrature_rule_1D(degree=1)
        
        @jit
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            internalPotential = self.compute_strain_energy(U, self.internalVariables)
            loadPotential = compute_traction_potential_energy(self.mesh, U, quadRule, self.mesh.sideSets['right'], right_traction_func)
            loadPotential += compute_traction_potential_energy(self.mesh, U, quadRule, self.mesh.sideSets['top'], top_traction_func)
            return internalPotential + loadPotential
        
        with Timer(name="NewtonSolve"):
            Uu = newton_solve(objective, dofManager.get_unknown_values(U))

            U = dofManager.create_field(Uu, Ubc)

        # exact solution
        modulus1 = (1.0 - self.nu**2)/self.E
        modulus2 = -self.nu*(1.0+self.nu)/self.E
        UExact = np.column_stack( ((modulus1*sigma11 + modulus2*sigma22)*self.mesh.coords[:,0],
                                   (modulus2*sigma11 + modulus1*sigma22)*self.mesh.coords[:,1]) )

        self.assertArrayNear(U, UExact, 14)


if __name__ == '__main__':
    TestFixture.unittest.main()
