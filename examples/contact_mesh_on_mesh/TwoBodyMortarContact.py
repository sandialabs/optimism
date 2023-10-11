from jax import numpy as np
from optimism import EquationSolver as EqSolver
from optimism import Objective
#from optimism.ConstrainedObjective import ConstrainedObjective
from optimism.ConstrainedObjective import ConstrainedQuasiObjective # in case of friction
from optimism import VTKWriter
from optimism import AlSolver

from optimism import Mesh
from optimism import Mechanics
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager
from optimism.contact import MortarContact

from optimism.material import Neohookean as Material
from optimism.test.MeshFixture import MeshFixture
from optimism.Timer import Timer

props = {'elastic modulus': 10.0,
         'poisson ratio': 0.25}

materialModel = Material.create_material_model_functions(props)

N = 4
M = 35

settings = EqSolver.get_settings(use_incremental_objective=False,
                                 min_tr_size=1e-14,
                                 tol=1e-7)

alSettings = AlSolver.get_settings(max_gmres_iters=100,
                                   num_initial_low_order_iterations=11,
                                   use_second_order_update=False,
                                   penalty_scaling = 1.1,
                                   target_constraint_decrease_factor=0.6,
                                   tol=2e-7)

def write_output(mesh, dofManager, U, step):
    plotName = get_output_name(step)
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)
    bcs = np.array(dofManager.isBc, dtype=int)
    writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)
    writer.write()

    
def get_output_name(N):
    return 'two-'+str(N).zfill(3)


def solve(Uu, energy_func, constraint_func, 
          update_params_func,
          write_output_func,
          numSteps, settings, alSettings,
          initialPenalty=4.0):

    step=0
    p = update_params_func(step, Uu, Objective.Params())

    c = constraint_func(Uu, p)
    kappa0 = initialPenalty * np.ones_like(c)
    lam0 = 1e-4*np.abs(kappa0*c)
        
    objective = ConstrainedQuasiObjective(energy_func,
                                          constraint_func,
                                          Uu,
                                          p,
                                          lam0,
                                          kappa0)

    write_output_func(step, Uu, p, objective.lam)
    
    for step in range(1,numSteps+1):
        print('\n------------ LOAD STEP', step, '------------\n')
                
        count=0
        def iteration_plot(Uu, p):
            nonlocal count
            count=count+1
            return

        residuals=[]
        def subproblem_residual(Uu, obj):
            errorNorm = np.linalg.norm(obj.total_residual(Uu))
            residuals.append(errorNorm)
            #print('error = ', errorNorm)
            with open('contact_residuals.'+str(count)+'.npz', 'wb') as file:
                np.savez(file,
                         data=np.array(residuals))
            
        p = update_params_func(step, Uu, p)
        Uu = AlSolver.augmented_lagrange_solve(objective, Uu, p, alSettings, settings, callback=iteration_plot, sub_problem_callback=subproblem_residual)
        write_output_func(step, Uu, p, objective.lam)


def get_ubcs(p, mesh, dofManager):    
    V = np.zeros(dofManager.ids.shape)
    return dofManager.get_bc_values(V)


def create_field(Uu, p, mesh, dofManager):
    return dofManager.create_field(Uu,
                                   get_ubcs(p, mesh, dofManager))


class ContactArch(MeshFixture):

    def setUp(self):
        w1 = 0.03
        w2 = 0.18
        archRadius1 = 1.5
        archRadius2 = archRadius1 + w1 + w2
        self.initialTop = archRadius2 + w2
                
        m1 = self.create_arch_mesh_disp_and_edges(N, M,
                                                  w1, archRadius1, 0.5*w1,
                                                  setNamePostFix='1')
        m2 = self.create_arch_mesh_disp_and_edges(N, M,
                                                  w2, archRadius2, 0.5*w2,
                                                  setNamePostFix='2')

        self.mesh, self.U = Mesh.combine_mesh(m1, m2)
        order=2
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=order, copyNodeSets=True)
        self.U = np.zeros(self.mesh.coords.shape)

        sideA = self.mesh.sideSets['top1']
        sideB = self.mesh.sideSets['bottom2']
        
        self.segmentConnsA = MortarContact.get_facet_connectivities(self.mesh, sideA)
        self.segmentConnsB = MortarContact.get_facet_connectivities(self.mesh, sideB)
        
        pushNodes = self.mesh.nodeSets['top2']

        triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        fs = FunctionSpace.construct_function_space(self.mesh,
                                                    triQuadRule)
        
        ebcs = [EssentialBC(nodeSet='left1', component=0),
                EssentialBC(nodeSet='left1', component=1),
                EssentialBC(nodeSet='right1', component=0),
                EssentialBC(nodeSet='right1', component=1),
                EssentialBC(nodeSet='left2', component=0),
                EssentialBC(nodeSet='left2', component=1),
                EssentialBC(nodeSet='right2', component=0),
                EssentialBC(nodeSet='right2', component=1)]
        
        self.dofManager = DofManager(fs, self.U.shape[1], ebcs)
        
        self.mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                              'plane strain',
                                                              materialModel)
        stateVars = self.mechFuncs.compute_initial_state()
        
        def energy_func(Uu, lam, p):
            U = create_field(Uu, p, self.mesh, self.dofManager)
            return self.mechFuncs.compute_strain_energy(U, stateVars)
        
        self.energy_func = energy_func

        def constraint_func(Uu, p):
            U = create_field(Uu, p, self.mesh, self.dofManager)
            yHeight = p[0]

            neighborList = p[1]
            nodalGapField = MortarContact.assemble_area_weighted_gaps(self.mesh.coords, U, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal)

            pushCoords = self.mesh.coords[pushNodes,1]+U[pushNodes,1]
            pushOverlap = yHeight - pushCoords

            return np.hstack([nodalGapField, pushOverlap])
                    
        self.constraint_func = constraint_func
            
        
    def run(self):
        numSteps = 40
        loadMag = 1.4

        maxContactNeighbors=4
        searchFrequency=1
        
        def create_field_func(Uu, p):
            return create_field(Uu, p, self.mesh, self.dofManager)

        
        def update_params_func(step, Uu, p):
            x = self.initialTop - step * loadMag / numSteps
            p = Objective.param_index_update(p, 0, x)

            if step%searchFrequency==0:
                U = create_field_func(Uu, p)
                neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, U, maxContactNeighbors)
                p = Objective.param_index_update(p, 1, neighborList)
            return p
        
        
        def write_output_func(step, Uu, p, lam):
            U = create_field_func(Uu, p)
            write_output(self.mesh, self.dofManager, U, step)


        Uu = self.dofManager.get_unknown_values(self.U)
        solve(Uu, self.energy_func, self.constraint_func,
              update_params_func,
              write_output_func,
              numSteps,
              settings, alSettings,
              initialPenalty = 25.0)
        

app = ContactArch()
app.setUp()
with Timer(name="AppRun"):
    app.run()