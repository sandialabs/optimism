import jax
import jax.numpy as np
import h5py as h5p
import numpy as onp
from scipy import sparse

from optimism import EquationSolver_Immersed_2 as EqSolver
from optimism import FunctionSpace
from optimism.material import Neohookean as MatModel
from optimism import Mechanics
from optimism import Mesh
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager
from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import QuadratureRule
from optimism.Timer import Timer
from optimism import VTKWriter
from optimism.test.MeshFixture import MeshFixture
from optimism.ReadExodusMesh import read__mesh


useNewton=False

if useNewton:
    solver = EqSolver.newton
else:
    solver = EqSolver.trust_region_minimize

class EulerBeam(MeshFixture):

    def setUp(self):
        # No mesh generation. We are reading the mesh from MORIS, in addition to the extraction operator.
        T_matrix_file = h5p.File('./Extraction_Buckling_Optimism_C.h5','r')
        T = np.transpose(np.array(T_matrix_file['T']))
        print(np.shape(T))
        self.w = 0.25
        #self.Tm = np.identity(715)
        self.Tm = T
        self.Tmn = onp.array(self.Tm)
        #h = 1.5
        #N = 5
        #M = 45
        #mesh, _ = self.create_mesh_and_disp(N, M, [-self.w/2,self.w/2], [0., h], lambda x: None) # Mesh set-up
        #mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, copyNodeSets=False) # hp-refinement
        #nodeSets = Mesh.create_nodesets_from_sidesets(mesh) # BC Patches?
        #self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets) # Integrate BC Patches ?
        mesh = read__mesh('./foreground_mesh_optimism_buckle_C.exo')
        nodeSets = Mesh.create_nodesets_from_sidesets(mesh)
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        

        # No change here.
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2) # Quadrature
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)

        ebcs = [EssentialBC(nodeSet='iside_b0_0_b1_1', component=0),
                EssentialBC(nodeSet='iside_b0_0_b1_2', component=0),
                EssentialBC(nodeSet='iside_b0_0_b1_2', component=1)]

        # Replace EBCs with empty array, because we are enforcing dirichlet BCs via Penalty.
        self.dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=[]) 
        
        # Quad
        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2) # quad

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}
        materialModel = MatModel.create_material_model_functions(props) 

        self.bvpFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                             mode2D="plane strain",
                                                             materialModel=materialModel) # Potential stuff again?

        #self.compute_bc_reactions = jax.jit(jax.grad(self.energy_from_bcs, 1)) # ??? gradient of potential

        self.trSettings = EqSolver.get_settings(max_trust_iters=100, t1=0.4, t2=1.5, eta1=1e-5, eta2=0.2, eta3=0.8, over_iters=500) # Solver

        self.outputForce = [0.0]
        self.outputDisp = [0.0]

        # Hit displacement with extraction operator in the main run function, so that the mapped displacement is only passed to all these.

    def energy_function_from_full_field(self, U, p):
        # Add penalty here. Replace with correct sideset name.
        
        internalVariables = p[1]
        strainEnergy = self.bvpFuncs.compute_strain_energy(U, internalVariables)
        F = p[0]

        loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_1'],
                                                                    lambda x, n: np.array([0.0, -F/self.w]))
        displacementPotential1 = Mechanics.compute_displacement_penalty(self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_2'],
                                                                    lambda x, n: np.array([0.0, 0.0]),0,100)
        displacementPotential2 = Mechanics.compute_displacement_penalty(self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_2'],
                                                                    lambda x, n: np.array([0.0, 0.0]),1,100)
        displacementPotential3 = Mechanics.compute_displacement_penalty(self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_1'],
                                                                    lambda x, n: np.array([0.0, 0.0]),0,100)
                                                                
        
       
       
        return strainEnergy + loadPotential + displacementPotential1 + displacementPotential2 +  displacementPotential3  # potential to minimize...
    
    # This one, not sure, but not required in the main forward solve.

    def energy_from_bcs(self, Uu, Ubc, p):
        U = self.dofManager.create_field(Uu, Ubc)
        return self.energy_function_from_full_field(U, p) # adding BC contributions

    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        return self.energy_function_from_full_field(U, p)

    def assemble_sparse(self, Uu, p):
        # No problem here. Just hit with extraction operator after the global stiffness matrix a.k.a. preconditioner inv computed.
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        elementStiffnesses =  self.bvpFuncs.compute_element_stiffnesses(U, internalVariables)
        d = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.fs.mesh.conns,
                                                                      self.dofManager).todense() # Background
        
        prec = sparse.csc_matrix(np.dot(np.dot(np.transpose(self.Tmn),d),self.Tmn))
        
        
        return prec

    def write_output(self, Uu, p, step):
        U = self.create_field(Uu, p)
        plotName = 'euler_column_immersed_2_buckle_optimism_C_clip-'+str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        Ubc = self.get_ubcs(p)
        internalVariables = p[1]
        #rxnBc = self.compute_bc_reactions(Uu, Ubc, p)
        #reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
        #writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

        energyDensities, stresses = self.bvpFuncs.compute_output_energy_densities_and_stresses(U, internalVariables)
        cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, energyDensities)
        cellStresses = FunctionSpace.project_quadrature_field_to_element_field(self.fs, stresses)
        writer.add_cell_field(name='strain_energy_density',
                              cellData=cellEnergyDensities,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)
        writer.add_cell_field(name='piola_stress',
                              cellData=cellStresses,
                              fieldType=VTKWriter.VTKFieldType.TENSORS)

        writer.write()

        force = p[0]
        #force2 = np.sum(reactions[:,1])
        print("applied force, reaction", force)
        disp = np.max(np.abs(U[self.mesh.sideSets['iside_b0_0_b1_1'],1]))
        print('Max Displacement')
        print(disp)
        self.outputForce.append(float(force))
        self.outputDisp.append(float(disp))

        with open('column_Fd_immersed.npz','wb') as f:
            np.savez(f, force=np.array(self.outputForce), displacement=np.array(self.outputDisp))


    def get_ubcs(self, p):
        V = np.zeros(self.mesh.coords.shape)
        return self.dofManager.get_bc_values(V) # No issues.


    def create_field(self, Uu, p):
            return self.dofManager.create_field(Uu, self.get_ubcs(p)) 
    



    def run(self):
        #T_matrix_file = h5p.File('./Extraction.h5','r')
        #T = np.transpose(np.array(T_matrix_file['T']))
        # Now Uu is defined in the background.
        #Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        #print(np.shape(self.Tm)[0])
        Uu = np.zeros(np.shape(self.Tm)[1])
        print('DoFs')
        print(np.shape(self.Tm)[0])
        force = 0.0
        ivs = self.bvpFuncs.compute_initial_state()
        p = Objective.Params(force, ivs)

        precondStrategy = Objective.PrecondStrategy(self.assemble_sparse)
        
        # Project to foreground
        objective = Objective.Objective(self.energy_function, np.dot(self.Tm,Uu), p, self.Tm, precondStrategy)

        #Project to foreground

        self.write_output(np.dot(self.Tm,Uu), p, step=0)

        steps = 5
        maxForce = 0.02
        for i in range(1, steps):
            key = jax.random.key(0)
            init_guess = Uu+jax.random.uniform(key=key,shape=np.shape(Uu)[0],minval=0,maxval=0.001)
            print('--------------------------------------')
            print('LOAD STEP ', i)
            force += maxForce/steps
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess, diff = EqSolver.nonlinear_equation_solve(objective, init_guess, p, self.Tm, self.trSettings, solver_algorithm=solver)
            # Project to foreground
            self.write_output(np.dot(self.Tm,Uu), p, i)
            dst = np.linalg.solve(np.dot(np.transpose(self.Tm),(self.Tm)),(np.dot(np.transpose(self.Tm),np.dot(self.Tm,Uu))))
            print("Norm Value")
            Tt = np.transpose(self.Tm)
            # Test whether solver is in background space or not.
            print(np.linalg.norm((dst)-Uu))
            # Check the scaling
            print(np.linalg.norm(diff-np.ones(np.shape(self.Tm)[1])))


        unload = False
        if unload:
            for i in range(steps, 2*steps - 1):
                print('--------------------------------------')
                print('LOAD STEP ', i)
                force -= maxForce/steps
                p = Objective.param_index_update(p, 0, force)
                Uu,  solverSuccess = EqSolver.nonlinear_equation_solve(objective, Uu, p, self.Tm, self.trSettings, solver_algorithm=solver)

                self.write_output(Uu, p, i)

app = EulerBeam()
app.setUp()
with Timer(name="AppRun"):
    app.run()

