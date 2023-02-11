from optimism.JaxConfig import *

from optimism import AlSolver
from optimism.ConstrainedObjective import ConstrainedObjective
from optimism.ConstrainedObjective import ConstrainedQuasiObjective
from optimism import EquationSolver as EqSolver
from optimism import Mesh
from optimism.material import LinearElastic
from optimism.contact import Friction
from optimism.contact import Levelset
from optimism.contact import LevelsetConstraint
from optimism.test.MeshFixture import MeshFixture, defaultProps, unittest
from optimism import Mechanics
from optimism import FunctionSpace
from optimism import QuadratureRule

props = {'elastic modulus': 1.0,
          'poisson ratio': 0.25}

materialModel = LinearElastic.create_material_model_functions(props)

materialModels = {'block': materialModel}

settings = EqSolver.get_settings()
alSettings = AlSolver.get_settings()
frictionParams = Friction.Params(mu = 0.3, sReg = 1e-4)

class TestLevelsetContactConstraint(MeshFixture):
    
    def setUp(self):

        self.Nx = 5
        self.Ny = 5
        self.xRange = [0.,1.]
        self.yRange = [0.,1.]

        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny,
                                                      self.xRange, self.yRange,
                                                      lambda x : 1e-14*x)
        
        quadratureRule2d = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(self.mesh, quadratureRule2d)
        
        EBCs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='left', component=1),
                FunctionSpace.EssentialBC(nodeSet='right', component=0),
                FunctionSpace.EssentialBC(nodeSet='right', component=1)]
        
        self.dofManager = FunctionSpace.DofManager(fs, dim=2, EssentialBCs=EBCs)
        
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(2)
        self.edges = self.mesh.sideSets['top']

        self.Uu = self.dofManager.get_unknown_values(self.U)
        self.Ubc = self.dofManager.get_bc_values(self.U)

        self.mechFuncs = Mechanics.create_multi_block_mechanics_functions(fs,
                                                                          'plane strain',
                                                                          materialModels)
        
        stateVars = self.mechFuncs.compute_initial_state()
        
        self.energy_func = lambda Uu,p : \
            self.mechFuncs.compute_strain_energy(self.dofManager.create_field(Uu, self.Ubc), stateVars)
        
        self.constraint_func = lambda levelset, Uu: LevelsetConstraint. \
            compute_levelset_constraints(levelset,
                                         self.dofManager.create_field(Uu, self.Ubc),
                                         mesh=self.mesh,
                                         quadRule=self.quadRule,
                                         edges=self.edges).ravel()
           
        
    def test_compute_all_positive_constraints_for_far_away_levelset(self):
        levelset = partial(Levelset.sphere, xLoc=0.5, yLoc=2.0, R=0.2)
        constraints=self.constraint_func(levelset, self.Uu)
        self.assertTrue( np.all( constraints > 0.0 ) )

    def test_some_positive_some_negative_constraints_for_small_sphere_on_edge(self):
        levelset = partial(Levelset.sphere, xLoc=0.5, yLoc=1.0, R=0.2)
        constraints=self.constraint_func(levelset, self.Uu)

        self.assertFalse( np.all( constraints > 0.0 ) )
        self.assertFalse( np.all( constraints < 0.0 ) )

    def test_solve(self):
        levelset = partial(Levelset.sphere, xLoc=0.5, yLoc=1.05, R=0.2)

        constraint_func = lambda x,p: self.constraint_func(levelset, x)
        tol = 1e-4

        #MeshPlot.plot_mesh_with_field(self.mesh, self.U, fast=False, direction=1, plotName='test.png')

        p = None
        lam0 = 0.0*constraint_func(self.Uu, p)
        kappa0 = 5 * np.ones(lam0.shape)
        objective = ConstrainedObjective(self.energy_func,
                                         constraint_func,
                                         self.Uu,
                                         p,
                                         lam0,
                                         kappa0)

        self.Uu = AlSolver.augmented_lagrange_solve(objective, self.Uu, p, alSettings, settings, useWarmStart=False)
        self.U = self.dofManager.create_field(self.Uu, self.Ubc)
        
        #MeshPlot.plot_mesh_with_field(self.mesh, self.U, fast=False, direction=1, plotName='test.png')

    def test_friction(self):

        initialSurfacePoints = LevelsetConstraint. \
            compute_contact_point_coordinates(self.dofManager.create_field(self.Uu, self.Ubc),
                                              self.mesh,
                                              self.quadRule,
                                              self.edges)
        
        initialLevelsetCenter = np.array([0.5, 1.05])
        
        levelset = partial(Levelset.sphere,
                           xLoc=initialLevelsetCenter[0],
                           yLoc=initialLevelsetCenter[1],
                           R=0.2)

        levelsetMotion = np.array([0.2, 0.0])

        energy_func = lambda Uu,lam,p: self.energy_func(Uu,p) + \
            LevelsetConstraint.compute_friction_potential(self.dofManager.create_field(Uu, self.Ubc),
                                                          initialSurfacePoints,
                                                          levelsetMotion,
                                                          lam,
                                                          self.mesh,
                                                          self.quadRule,
                                                          self.edges,
                                                          frictionParams)
                                                          
                                                          
        constraint_func = lambda x,p: self.constraint_func(levelset, x)
        tol = 1e-4

        #MeshPlot.plot_mesh_with_field(self.mesh, self.U, fast=False, direction=1, plotName='test.png')
        p = None
        lam0 = 0.0*constraint_func(self.Uu, p)
        kappa0 = 5 * np.ones(lam0.shape)
        
        objective = ConstrainedQuasiObjective(energy_func, constraint_func,
                                              self.Uu,
                                              p,
                                              lam0,
                                              kappa0)

        self.Uu = AlSolver.augmented_lagrange_solve(objective, self.Uu, p, alSettings, settings, useWarmStart=False)
        self.U = self.dofManager.create_field(self.Uu, self.Ubc)


if __name__ == '__main__':
    unittest.main()

    
