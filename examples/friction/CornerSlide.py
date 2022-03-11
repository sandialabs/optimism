from optimism.JaxConfig import *


from optimism.material import Neohookean
from optimism import Mechanics
from optimism import FunctionSpace
from optimism import EquationSolver as EqSolver
from optimism import VTKWriter
from optimism import AlSolver
from optimism import QuadratureRule
from optimism import Mesh
from optimism import TractionBC

from optimism.Mesh import EssentialBC
from optimism.Mesh import DofManager
from optimism.contact import PenaltyContact
from optimism.contact import Friction
from optimism.contact import Levelset
from optimism.contact import LevelsetConstraint
from optimism.Timer import Timer
from optimism import Objective
from optimism.ConstrainedObjective import ConstrainedObjective
from optimism.ConstrainedObjective import ConstrainedQuasiObjective

from optimism.test.MeshFixture import MeshFixture


#kappa = 10
#nu = 0.3
#mu = 1.5 * kappa * ( 1-2*nu ) / ( 1+nu )

props = {'elastic modulus': 10.0,
         'poisson ratio': 0.3}

materialModel = Neohookean.create_material_model_functions(props)

factor = 1.0 #100.0

settings = EqSolver.get_settings(max_cg_iters=150,
                                 tr_size=1e-3,
                                 min_tr_size=1e-13,
                                 tol=factor*5e-10,
                                 use_incremental_objective=True)

alSettings = AlSolver.get_settings(max_gmres_iters=200,
                                   num_initial_low_order_iterations=10000,
                                   penalty_scaling=1.1,
                                   tol=factor*1e-9)

frictionParams = Friction.Params(mu = 0.3, sReg = 1e-6)


class CornerSlide(MeshFixture):
    
    def setUp(self):
        self.mesh, self.U = self.create_mesh_and_disp(5,5, [0,1], [0,1], lambda x : 0*x)
        
        self.edges = np.vstack((self.mesh.sideSets['left'], self.mesh.sideSets['bottom']))
        
        EBCs = []
        self.dofManager = DofManager(self.mesh, self.U.shape, EBCs)
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(2)

        self.Ubc = self.dofManager.get_bc_values(self.U)
        
        initialSurfacePoints = LevelsetConstraint. \
            compute_contact_point_coordinates(self.U,
                                              self.mesh,
                                              self.quadRule,
                                              self.edges)
        

        triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(self.mesh,
                                                    triQuadRule)

        self.mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                              'plane strain',
                                                              materialModel)
        
        self.state = self.mechFuncs.compute_initial_state()
        
        def energy_func(Uu, lam, p):
            U = self.create_field(Uu, p)

            tractionRelMag = p[0][0]
            traction = 1e-5
            
            rightPushEnergy = 0 #TractionBC.compute_traction_potential_energy(self.mesh, U, self.quadRule, self.mesh.sideSets['right'], lambda x: np.array([-0.3, 0.0]))
            topPushEnergy = TractionBC.compute_traction_potential_energy(self.mesh, U, self.quadRule, self.mesh.sideSets['top'], lambda x: np.array([-tractionRelMag*traction, -traction]))
            mechanicalEnergy = self.mechFuncs.compute_strain_energy(U, self.state)

            levelsetMotion = np.array([0.0, 0.0])
            
            frictionEnergy = LevelsetConstraint.compute_friction_potential(U,
                                                                           initialSurfacePoints,
                                                                           levelsetMotion,
                                                                           lam,
                                                                           self.mesh,
                                                                           self.quadRule,
                                                                           self.edges,
                                                                           frictionParams)
            
            return mechanicalEnergy + rightPushEnergy + topPushEnergy + frictionEnergy
        self.energy_func = energy_func
        
        def constraint_func(Uu, p):
            cornerLoc = p[0][1:3]
            levelset = partial(Levelset.corner, xLoc=cornerLoc[0], yLoc=cornerLoc[1])
            U = self.create_field(Uu, p)
            return PenaltyContact.evaluate_contact_constraints(levelset, U, self.mesh, self.quadRule, self.edges).ravel()
        self.constraint_func = constraint_func
        

    @partial(jit, static_argnums=0)
    def create_field(self, Uu, p):
        return self.dofManager.create_field(Uu, self.get_ubcs(p))

    
    def get_ubcs(self, p):
        return self.Ubc

    
    def plot_solution(self, dispField, plotName, p):
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=dispField,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        
        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs',
                               nodalData=bcs,
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)
        
        writer.write()


    def get_output_name(self, N):
        return 'corner-'+str(N).zfill(3)


    #@partial(Timer, name='Total')
    def run(self):

        Uu = self.dofManager.get_unknown_values(self.U)
        p = Objective.Params(np.array([0.0, -0.1, -0.1]))

        lam0 = 0.0 * self.constraint_func(Uu, p)
        penalty = 2.5
        kappa0 = penalty * np.ones(lam0.shape)

        hess_func = jit(hessian(lambda Uu,p: self.energy_func(Uu,lam0,p)))
        
        objective = ConstrainedQuasiObjective(self.energy_func,
                                              self.constraint_func,
                                              Uu,
                                              p,
                                              lam0,
                                              kappa0)
                
        self.plot_solution(self.create_field(Uu, p),
                           self.get_output_name(0), p)

        outputDisp = []
        outputForce = []
        
        N = 50
        for i in range(N):
            print("")
            print("----- LOAD STEP " + str(i+1) + " -----")
            
            forceRatio = 0.29025 + i*0.0003
            p = Objective.Params(ops.index_update(p[0], 0, forceRatio))
            print('force ratio = ', forceRatio)
            
            Uu = AlSolver.augmented_lagrange_solve(objective, Uu, p, alSettings, settings)

            outputForce.append(forceRatio)
            outputDisp.append( np.average(self.create_field(Uu, p)[:,0]) )
            
            with open('force_disp.npz', 'wb') as f:
                np.savez(f,
                         disp=np.array(outputDisp),
                         force=np.array(outputForce))
            
            self.plot_solution(self.create_field(Uu, p),
                               self.get_output_name(i+1), p)

                
app = CornerSlide()
app.setUp()
app.run()

