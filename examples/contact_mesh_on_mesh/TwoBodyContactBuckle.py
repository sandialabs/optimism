from TwoBodyContact import *

from functools import partial

from optimism.material import Neohookean as Material
from optimism.contact import Levelset
from optimism.contact import PenaltyContact

props = {'elastic modulus': 10.0,
         'poisson ratio': 0.25}

materialModel = Material.create_material_model_functions(props)

N = 7
M = 85

#N = 3
#M = 12

h = 1.5 / M # approximate element size
smoothingDistance = 1e-1 * h
closest_distance_func = Contact.compute_closest_distance_to_each_side_smooth


settings = EqSolver.get_settings(use_incremental_objective=False,
                                 max_trust_iters=300,
                                 tr_size=0.25,
                                 min_tr_size=1e-15,
                                 tol=5e-8)

alSettings = AlSolver.get_settings(max_gmres_iters=75,
                                   num_initial_low_order_iterations=10,
                                   use_second_order_update=True,
                                   penalty_scaling = 1.05,
                                   target_constraint_decrease_factor=0.5,
                                   tol=2e-7)

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
        
        self.sphereRadius = 0.25
        self.sphereCenter = 0.5-self.sphereRadius
                
        m1 = \
            self.create_arch_mesh_disp_and_edges(N, M,
                                                 w1, archRadius1, 0.5*w1,
                                                 setNamePostFix='1')

        m2 = \
            self.create_arch_mesh_disp_and_edges(N, M,
                                                 w2, archRadius2, 0.5*w2,
                                                 setNamePostFix='2')

        self.mesh, self.U = Mesh.combine_mesh(m1, m2)
        
        self.edges1 = self.mesh.sideSets['top1']
        self.edges2 = self.mesh.sideSets['bottom2']
        self.selfContactEdges = self.mesh.sideSets['bottom1']
        
        pushNodes = self.mesh.nodeSets['top2']

        popNodes = self.mesh.nodeSets['bottom1']
        popNodes = popNodes[ np.where( np.abs(self.mesh.coords[popNodes,0]) < 1.0 ) ]

        self.sphereLevelset = partial(Levelset.sphere,
                                      xLoc=0.0,
                                      yLoc=self.sphereCenter,
                                      R=self.sphereRadius)
        
        triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
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
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(4)
        
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
            interactionList = p[1][0]
            selfInteractionList = p[1][1]
            contactDists = closest_distance_func(self.mesh, U, self.quadRule, interactionList, self.edges2, smoothingDistance).ravel()

            #selfInteractionDists = closest_distance_func(self.mesh, U, self.quadRule, selfInteractionList, self.selfContactEdges, smoothingDistance).ravel()
            
            # try to avoid collisions with directly neighboring edges
            #selfInteractionDists = np.where(selfInteractionDists < -h / 8.0, 1.0, selfInteractionDists)
            
            pushCoords = self.mesh.coords[pushNodes,1]+U[pushNodes,1]
            pushOverlap = yHeight - pushCoords
            
            popDists = self.mesh.coords[popNodes,1]+U[popNodes,1]
            popOverlap = popDists - self.sphereCenter
            
            return np.hstack([contactDists, pushOverlap, popOverlap])
                    
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
                interactionList = Contact.get_potential_interaction_list(self.edges1, self.edges2,
                                                                         self.mesh, U, maxContactNeighbors)

                selfInteractionList = Contact.get_potential_interaction_list(self.selfContactEdges,
                                                                             self.selfContactEdges,
                                                                             self.mesh, U, maxContactNeighbors)

                #print('self ilist = ', selfInteractionList)
                #print('size1 = ', selfInteractionList.shape, self.selfContactEdges.shape)
                
                def is_another_neighbor(qneighbor, resident):
                    return np.where( np.any(qneighbor!=resident), True, False )

                
                def filter_edge_neighbors(eneighbors, resident):
                    isNeighbor = jax.vmap(is_another_neighbor, (0,None))(eneighbors, resident) #[is_another_neighbor(q, resident) for q in eneighbors]
                    return eneighbors[np.where(isNeighbor)]
                
                selfInteractionList = np.array( [filter_edge_neighbors(eneighbors, self.selfContactEdges[e]) for e, eneighbors in enumerate(selfInteractionList) ] )
                
                p = Objective.param_index_update(p, 1, (interactionList, selfInteractionList) )
            return p
        
        
        def write_output_func(step, Uu, p, lam):
            U = create_field_func(Uu, p)
            write_output(self.mesh, self.dofManager, U, p, self.mechFuncs, step, [[0.0,self.sphereCenter], [0.0,p[0]]], [self.sphereRadius, 1.])


        def write_debug_output_func(subStep, Uu, p, lam):
            U = create_field_func(Uu, p)
            write_debug_output(self.mesh, self.quadRule, self.edges2, U, p, subStep)


        Uu = self.dofManager.get_unknown_values(self.U)
        solve(Uu, self.energy_func, self.constraint_func,
              update_params_func,
              write_output_func,
              write_debug_output_func,
              numSteps,
              settings, alSettings,
              initialMultiplier = 25.0)
        

app = ContactArch()
app.setUp()
with Timer(name="AppRun"):
    app.run()

