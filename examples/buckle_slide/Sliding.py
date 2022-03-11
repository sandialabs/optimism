from optimism.JaxConfig import *

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import VTKWriter
from optimism import AlSolver
from optimism import QuadratureRule
from optimism.material import Neohookean
from optimism import Mesh
from optimism import Mechanics

from optimism.Mesh import EssentialBC
from optimism.Mesh import DofManager
from optimism.contact import PenaltyContact
from optimism.contact import Levelset
from optimism.Timer import Timer
from optimism import Objective
from optimism.ConstrainedObjective import ConstrainedObjective
from optimism.material import Neohookean

from optimism.test.MeshFixture import MeshFixture

import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=2"

props = {'elastic modulus': 10.0,
         'poisson ratio': 0.3}

materialModel = Neohookean.create_material_model_functions(props)

# compute_free_energy_density = lambda strain, Q : neohookean_3D_energy_density(strain, Q, props)

secondOrderUpdate=True

settings = EqSolver.get_settings(max_cg_iters=100, tol=6e-13)
alSettings = AlSolver.get_settings(max_gmres_iters=200,
                                   tol=1e-12,
                                   use_second_order_update=secondOrderUpdate,
                                   #use_second_order_update=False,
                                   num_initial_low_order_iterations=2) 
#use_newton_only = True

class Sliding(MeshFixture):

    def setUp(self):
        self.w = 0.02
        self.curveLength = 2.0
        self.warpDistance = 1e-8
        self.ballRadius = self.curveLength/7.0
        self.initialBallX = -0.45
        self.initialBallY = self.warpDistance + self.w + self.ballRadius + 0.2
        
        N = 6
        M = 101

        self.mesh, self.U = \
            self.create_cos_mesh_disp_and_edges(N, M,
                                                self.w,
                                                self.curveLength,
                                                self.warpDistance)

        self.edges = self.mesh.sideSets['top']
        
        EBCs = []
        EBCs.append(EssentialBC(nodeSet='left', field=0))
        EBCs.append(EssentialBC(nodeSet='left', field=1))
        EBCs.append(EssentialBC(nodeSet='right', field=0))
        EBCs.append(EssentialBC(nodeSet='right', field=1))
        
        self.dofManager = DofManager(self.mesh, self.U.shape, EBCs)
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(2)

        triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(self.mesh,
                                                    triQuadRule)
        
        self.mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                              'plane strain',
                                                              materialModel)
                
        def energy_func(Uu, p):
            U = self.create_field(Uu, p)
            stateVars = p[1]
            return self.mechFuncs.compute_strain_energy(U, stateVars)
        
        self.energy_func = energy_func
        
        def constraint_func(Uu, p):
            sphereLoc = p[0][1:]
            levelset = partial(Levelset.sphere,
                               xLoc=sphereLoc[0],
                               yLoc=sphereLoc[1],
                               R=self.ballRadius)
            U = self.create_field(Uu, p)
            constraints = PenaltyContact.evaluate_contact_constraints(levelset,
                                                                      U,
                                                                      self.mesh,
                                                                      self.quadRule,
                                                                      self.edges)
            return constraints.ravel()
        
        self.constraint_func = constraint_func

        
    @partial(jit, static_argnums=0)
    def create_field(self, Uu, p):
        return self.dofManager.create_field(Uu, self.get_ubcs(p))

    
    def get_ubcs(self, p):
        dispX = p[0][0]
        V = np.zeros(self.U.shape)
        indexR = ops.index[self.mesh.nodeSets['right'],0]
        V = ops.index_update(V, indexR, -dispX)
        indexL = ops.index[self.mesh.nodeSets['left'],0]
        V = ops.index_update(V, indexL, dispX)
        return self.dofManager.get_bc_values(V)

    def plot_solution(self, dispField, plotName, p):
        params = p[0]
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=dispField,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        
        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs',
                               nodalData=bcs,
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)

        writer.add_sphere(params[1:], self.ballRadius)
        
        writer.write()


    def get_output_name(self, N):
        return 'sliding-'+str(N).zfill(3)


    def x_motion(self, step, N):
        xTotal = 1.1
        if step <= N:
            return xTotal * step / N
        else:
            return xTotal - 0.4 * (step-N) / N
    
    
    def y_motion(self, step, N):
        if step <= N:
            return 0.0
        else:
            return -0.4 * (step-N) / N
        
        
    def run(self):
        Uu = self.dofManager.get_unknown_values(self.U)
        stateVars = self.mechFuncs.compute_initial_state()
        
        p = Objective.Params(np.array([0.0, self.initialBallX, self.initialBallY]), stateVars)
        
        lam0 = 0.0 * self.constraint_func(Uu, p)
        kappa0 = 0.02 * np.ones(lam0.shape)

        objective = ConstrainedObjective(self.energy_func,
                                         self.constraint_func,
                                         Uu,
                                         p,
                                         lam0,
                                         kappa0)
        
        with Timer(name='Total'):
            N = 20
            loadMag = 0.1 #4.0

            self.plot_solution(self.create_field(Uu, p),
                               self.get_output_name(0), p)
            
            for i in range(2*N+1):
                print("")
                print("----- LOAD STEP " + str(i+1) + " -----")

                pushInDisp = loadMag
                sphereDispX = self.initialBallX + self.x_motion(i, N)
                sphereDispY = self.initialBallY + self.y_motion(i, N)
                p = Objective.param_index_update(p, 0, np.array([pushInDisp, sphereDispX, sphereDispY]))

                residuals = []
                def al_callback(Uu, p):
                    errorNorm = np.linalg.norm(objective.total_residual(Uu))
                    residuals.append(errorNorm)
                    print('error = ', errorNorm)
                    name =  'contact_residuals' if secondOrderUpdate else 'contact_residuals_first'
                    with open(name+'.'+str(i)+'.npz', 'wb') as file:
                        np.savez(file,
                                 data=np.array(residuals))

                    
                
                Uu = AlSolver.augmented_lagrange_solve(objective, Uu, p, alSettings, settings, callback = al_callback)

                self.plot_solution(self.create_field(Uu, p),
                                   self.get_output_name(i+1), p)

            
app = Sliding()
app.setUp()
app.run()
    

