from TwoBodyContact import *
from optimism.material import Neohookean as Material
#from optimism import J2Plastic as Material
from optimism.contact import Friction

props1 = {'elastic modulus': 0.25,
          'poisson ratio': 0.3}

props2 = {'elastic modulus': 10.0,
          'poisson ratio': 0.3}

materialModel1 = Material.create_material_model_functions(props1)
materialModel2 = Material.create_material_model_functions(props2)

materialModels = {'block1': materialModel1,
                  'block2': materialModel2,
                  'block3': materialModel1,
                  'block4': materialModel2}

N = 15
M = 15

h = 1.0 / (M-1) # approximate element size

smoothingDistance = 1.e-1 * h

closest_distance_func = Contact.compute_closest_distance_to_each_side_smooth

settings = EqSolver.get_settings(use_incremental_objective=True,
                                 max_trust_iters=500,
                                 tr_size=0.05,
                                 min_tr_size=1e-15,
                                 tol=1.e-7)

alSettings = AlSolver.get_settings(max_gmres_iters=200,
                                   num_initial_low_order_iterations=300,
                                   use_second_order_update=False,
                                   penalty_scaling = 1.01,
                                   target_constraint_decrease_factor=0.9,
                                   tol=2.e-7)

frictionParams1 = Friction.Params(mu = 0.25, sReg = 5e-6)
frictionParams2 = Friction.Params(mu = 0.65, sReg = 5e-6)
frictionParams3 = Friction.Params(mu = 0.45, sReg = 5e-6)


def get_ubcs(p, mesh, dofManager):    
    V = np.zeros(dofManager.ids.shape)

    topDispY = p[0]
    V = V.at[mesh.nodeSets['top4'], 1].set(topDispY)
    
    return dofManager.get_bc_values(V)


def create_field(Uu, p, mesh, dofManager):
    return dofManager.create_field(Uu,
                                   get_ubcs(p, mesh, dofManager))


class ContactArch(MeshFixture):

    def setUp(self):

        xRange = np.array([0.0, 1.0])
        yRange = np.array([0.0, 1.0])

        almostOne = 1.0 + 1e-7
        
        m1 = self.create_mesh_and_disp(N-1, M,
                                       xRange,
                                       yRange,
                                       lambda x:0.*x,
                                       setNamePostFix='1')
        
        m2 = self.create_mesh_and_disp(N+1, M+1,
                                       xRange,
                                       yRange+almostOne,
                                       lambda x:0.*x,
                                       setNamePostFix='2')

        m12 = Mesh.combine_mesh(m1, m2)

        m3 = self.create_mesh_and_disp(N, M-1,
                                       xRange,
                                       yRange+2.0*almostOne,
                                       lambda x:0.*x,
                                       setNamePostFix='3')

        m4 = self.create_mesh_and_disp(N+1, M,
                                       xRange,
                                       yRange+3.0*almostOne,
                                       lambda x:0.*x,
                                       setNamePostFix='4')
        
        m34 = Mesh.combine_mesh(m3, m4)

        self.mesh, self.U = Mesh.combine_mesh(m12, m34)
        
        self.e12 = self.mesh.sideSets['top1']
        self.e21 = self.mesh.sideSets['bottom2']

        self.e23 = self.mesh.sideSets['top2']
        self.e32 = self.mesh.sideSets['bottom3']
        
        self.e34 = self.mesh.sideSets['top3']
        self.e43 = self.mesh.sideSets['bottom4']

        self.nEdges12 = self.e21.shape[0]
        self.nEdges23 = self.e32.shape[0]
        self.nEdges34 = self.e43.shape[0]
        
        triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(self.mesh,
                                                    triQuadRule)
        
        ebcs = [EssentialBC(nodeSet='bottom1', component=0),
                EssentialBC(nodeSet='bottom1', component=1),
                EssentialBC(nodeSet='top4', component=0),
                EssentialBC(nodeSet='top4', component=1)]
        self.dofManager = DofManager(fs, self.U.shape[1], ebcs)

        self.quadRule = QuadratureRule.create_quadrature_rule_1D(4)
        
        self.mechFuncs = Mechanics.create_multi_block_mechanics_functions(fs,
                                                                          'plane strain',
                                                                          materialModels)

        stateVars = self.mechFuncs.compute_initial_state()

        
        def energy_func(Uu, lam, p):
            U = create_field(Uu, p, self.mesh, self.dofManager)

            lam = np.reshape(lam, (self.nEdges12+self.nEdges23+self.nEdges34, QuadratureRule.len(self.quadRule)))
            
            frictionDatas = p[3]
            
            frictionEnergy12 = Contact.compute_friction_potential(self.mesh, U,
                                                                  lam[:self.nEdges12],
                                                                  frictionParams1,
                                                                  self.quadRule, self.e21,
                                                                  *frictionDatas[0])

            frictionEnergy23 = Contact.compute_friction_potential(self.mesh, U,
                                                                  lam[self.nEdges12:self.nEdges12+self.nEdges23],
                                                                  frictionParams2,
                                                                  self.quadRule, self.e32,
                                                                  *frictionDatas[1])

            frictionEnergy34 = Contact.compute_friction_potential(self.mesh, U,
                                                                  lam[self.nEdges12+self.nEdges23:],
                                                                  frictionParams3,
                                                                  self.quadRule, self.e43,
                                                                  *frictionDatas[2])
            
            return self.mechFuncs.compute_strain_energy(U, stateVars) + frictionEnergy12 + frictionEnergy23 + frictionEnergy34
        
        self.energy_func = energy_func

        
        def constraint_func(Uu, p):
            U = create_field(Uu, p, self.mesh, self.dofManager)
            
            interactionLists = p[1]

            contactDists12 = closest_distance_func(self.mesh, U, self.quadRule, interactionLists[0], self.e21, smoothingDistance).ravel()
            contactDists23 = closest_distance_func(self.mesh, U, self.quadRule, interactionLists[1], self.e32, smoothingDistance).ravel()
            contactDists34 = closest_distance_func(self.mesh, U, self.quadRule, interactionLists[2], self.e43, smoothingDistance).ravel()
            return np.hstack([contactDists12, contactDists23, contactDists34])
                    
        self.constraint_func = constraint_func
            
        
    def run(self):

        numSteps = 200
        loadMag = 1.0

        maxContactNeighbors=4
        searchFrequency=1
        
        def create_field_func(Uu, p):
            return create_field(Uu, p, self.mesh, self.dofManager)

        
        def update_params_func(step, Uu, p):
            topDisp = -step * loadMag / numSteps
            
            p = Objective.param_index_update(p, 0, topDisp)
            
            if step%searchFrequency==0:
                U = create_field_func(Uu, p)
                
                interactionList12 = Contact.get_potential_interaction_list(self.e12, self.e21,
                                                                           self.mesh, U, maxContactNeighbors)
                interactionList23 = Contact.get_potential_interaction_list(self.e23, self.e32,
                                                                           self.mesh, U, maxContactNeighbors)
                interactionList34 = Contact.get_potential_interaction_list(self.e34, self.e43,
                                                                           self.mesh, U, maxContactNeighbors)
                interactionLists = (interactionList12, interactionList23, interactionList34)
                
                p = Objective.param_index_update(p, 1, interactionLists)

            fData12 = Contact.compute_closest_edges_and_field_weights(self.mesh, self.U,
                                                                      self.quadRule,
                                                                      p[1][0],
                                                                      self.e21)

            fData23 = Contact.compute_closest_edges_and_field_weights(self.mesh, self.U,
                                                                      self.quadRule,
                                                                      p[1][1],
                                                                      self.e32)

            fData34 = Contact.compute_closest_edges_and_field_weights(self.mesh, self.U,
                                                                      self.quadRule,
                                                                      p[1][2],
                                                                      self.e43)

            fData = (fData12, fData23, fData34)
            
            p = Objective.param_index_update(p, 3, fData)
                
            return p
        
        
        def write_output_func(step, Uu, p, lam):
            U = create_field_func(Uu, p)
            write_output(self.mesh, self.dofManager, U, p, self.mechFuncs, step)


        def write_debug_output_func(subStep, Uu, p, lam):
            U = create_field_func(Uu, p)
            write_debug_output(self.mesh, self.quadRule, self.e21, U, p, subStep)

        
        Uu = self.dofManager.get_unknown_values(self.U)
        solve(Uu, self.energy_func, self.constraint_func,
              update_params_func,
              write_output_func,
              write_debug_output_func,
              numSteps,
              settings, alSettings,
              initialMultiplier = 1000.0)
        

app = ContactArch()
app.setUp()
with Timer(name="AppRun"):
    app.run()

