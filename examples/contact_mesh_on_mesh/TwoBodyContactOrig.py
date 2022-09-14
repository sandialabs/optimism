from TwoBodyContact import *
from optimism import Neohookean as Material

props = {'elastic modulus': 10.0,
         'poisson ratio': 0.25}
materialModel = Material.create_material_model_functions(props)

N = 6
M = 75

smoothingDistance = 1e-1 * 1.5 / N
closest_distance_func = Contact.compute_closest_distance_to_each_side_smooth


settings = EqSolver.get_settings(use_incremental_objective=False,
                                 min_tr_size=1e-15,
                                 tol=1e-7)

alSettings = AlSolver.get_settings(max_gmres_iters=50,
                                   num_initial_low_order_iterations=5,
                                   use_second_order_update=True,
                                   penalty_scaling = 1.1,
                                   target_constraint_decrease_factor=0.5,
                                   tol=2e-7)


def get_ubcs(p, mesh, dofManager):    
    V = np.zeros(dofManager.ids.shape)
    dispY = p[0]

    V = V.at[mesh.nodeSets['left2'], 1].set(dispY)
    
    V = V.at[mesh.nodeSets['right2'], 1].set(dispY)
    
    bcVals = dofManager.get_bc_values(V)
    return bcVals


def create_field(Uu, p, mesh, dofManager):
    return dofManager.create_field(Uu,
                                   get_ubcs(p, mesh, dofManager))


class ContactArch(MeshFixture):

    def setUp(self):
        w = 0.07
        archRadius = 1.5
        self.initialTopDisp = 3.2
                
        m1 = \
            self.create_arch_mesh_disp_and_edges(N, M,
                                                 w, archRadius, 0.5*w,
                                                 setNamePostFix='1')

        mesh2, disp2 = \
            self.create_arch_mesh_disp_and_edges(N+1, M+1,
                                                 w, archRadius, 0.5*w,
                                                 setNamePostFix='2')

        R = -np.identity(2)
        coords2 = jax.vmap(lambda x: R@x)(mesh2.coords)
        coords2 = coords2.at[:,1].add(self.initialTopDisp)
        mesh2 = Mesh.mesh_with_coords(mesh2, coords2)

        self.mesh, self.U = Mesh.combine_mesh(m1, (mesh2,disp2) )
        
        self.edges1 = self.mesh.sideSets['top1']
        self.edges2 = self.mesh.sideSets['top2']
        
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
            interactionList = p[1][0]
            U = create_field(Uu, p, self.mesh, self.dofManager)
            dists = closest_distance_func(self.mesh, U, self.quadRule, interactionList, self.edges2, smoothingDistance).ravel()            
            return dists
        
        self.constraint_func = constraint_func
        
            
    def run(self):

        numSteps = 50
        loadMag = -4.0

        maxContactNeighbors=4
        searchFrequency=1
        
        def create_field_func(Uu, p):
            return create_field(Uu, p, self.mesh, self.dofManager)

        
        def update_params_func(step, Uu, p):
            xDisp = step*loadMag/numSteps
            p = Objective.param_index_update(p, 0, xDisp)
            if step%searchFrequency==0:
                U = create_field_func(Uu, p)
                interactionList = Contact.get_potential_interaction_list(self.edges1, self.edges2,
                                                                         self.mesh, U, maxContactNeighbors)
                p = Objective.param_index_update(p, 1, (interactionList,None))
            return p
        
        
        def write_output_func(step, Uu, p, lam):
            U = create_field_func(Uu, p)

            quadPoints, dists = get_contact_points_and_field(self.edges2, self.mesh, U, p, self.quadRule, smoothingDistance)

            numSpheres = quadPoints.shape[0] * quadPoints.shape[1]
            
            quadPoints = np.reshape(quadPoints, [numSpheres, 2])
            lam = np.reshape(lam, [numSpheres])

            write_output(self.mesh, self.dofManager, U, p, self.mechFuncs, step, quadPoints, lam)


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
              initialMultiplier = 20.)
        

app = ContactArch()
app.setUp()
with Timer(name="AppRun"):
    app.run()
    
