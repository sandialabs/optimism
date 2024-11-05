import jax
import jax.numpy as np
import numpy as onp
import h5py as h5p

from optimism import EquationSolver as EqSolver
from optimism import EquationSolverSubspace as SolverSubspace
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



class TractionArch(MeshFixture):

    def setUp(self):
        self.w = 0.3
        self.archRadius = 1.5
        self.ballRadius = self.archRadius/5.0
        self.initialBallLoc = self.archRadius + self.w + self.ballRadius
        N = 5
        M = 65
        
        #mesh, _ = \
        #    self.create_arch_mesh_disp_and_edges(N, M,
        #                                         self.w, self.archRadius, self.w)

        mesh = read__mesh('./foreground_mesh_shallow_arch_C.exo')
        #mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, copyNodeSets=False)
        nodeSets = Mesh.create_nodesets_from_sidesets(mesh)
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        
        ebcs = [EssentialBC(nodeSet='iside_b0_0_b1_3', component=0),
                EssentialBC(nodeSet='iside_b0_0_b1_3', component=1)]

        self.dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        self.pushArea = 0.0396*self.mesh.sideSets['iside_b0_0_b1_2'].shape[0]
        
        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}
        materialModel = MatModel.create_material_model_functions(props)

        self.bvpFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                             mode2D="plane strain",
                                                             materialModel=materialModel)

        def compute_energy_from_bcs(Uu, Ubc, p):
            U = self.dofManager.create_field(Uu, Ubc)
            internalVariables = p[1]
            strainEnergy = self.bvpFuncs.compute_strain_energy(U, internalVariables)
            F = p[0]
            loadPotential = Mechanics.compute_traction_potential_energy(
                self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_2'], 
                lambda x, n: np.array([0.0, -F/self.pushArea]))
            return strainEnergy + loadPotential
        
        self.compute_bc_reactions = jax.jit(jax.grad(compute_energy_from_bcs, 1))
        
        self.trSettings = EqSolver.get_settings(max_trust_iters=400, t1=0.4, t2=1.5, eta1=1e-8, eta2=0.2, eta3=0.8, over_iters=100)
        
        self.outputForce = [0.0]
        self.outputDisp = [0.0]


    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        strainEnergy = self.bvpFuncs.compute_strain_energy(U, internalVariables)
        F = p[0]
        loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_2'], 
                                                                    lambda x, n: np.array([0.0, -F/self.pushArea]))
        return strainEnergy + loadPotential

    
    def assemble_sparse(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        elementStiffnesses =  self.bvpFuncs.compute_element_stiffnesses(U, internalVariables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.fs.mesh.conns,
                                                                      self.dofManager)


    def write_output(self, Uu, p, step):
        U = self.create_field(Uu, p)
        plotName = 'arch_traction_C-'+str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        
        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        Ubc = self.get_ubcs(p)
        internalVariables = p[1]
        rxnBc = self.compute_bc_reactions(Uu, Ubc, p)
        reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
        writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

        energyDensities, stresses = self.bvpFuncs.\
            compute_output_energy_densities_and_stresses(U, internalVariables)
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
        force2 = np.sum(reactions[:,1])
        print("applied force, reaction", force, force2)
        disp = np.max(np.abs(U[self.mesh.nodeSets['iside_b0_0_b1_2'],1]))
        self.outputForce.append(float(force))
        self.outputDisp.append(float(disp))
        self.dispmax = disp
        self.forcev = force

        with open('arch_traction_Fd.npz','wb') as f:
            np.savez(f, force=np.array(self.outputForce), displacement=np.array(self.outputDisp))

            
    def get_ubcs(self, p):
        V = np.zeros(self.mesh.coords.shape)
        return self.dofManager.get_bc_values(V)

    
    def create_field(self, Uu, p):
            return self.dofManager.create_field(Uu, self.get_ubcs(p))

        
    def run(self):
        Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        force = 0.0
        ivs = self.bvpFuncs.compute_initial_state()
        p = Objective.Params(force, ivs)

        precondStrategy = Objective.PrecondStrategy(self.assemble_sparse)
        objective = Objective.Objective(self.energy_function, Uu, p, 1.0, precondStrategy)

        self.disp = np.array([])
        self.force = np.array([]) 

        self.write_output(Uu, p, step=0)
        
        steps = 40
        maxForce = 0.1
        for i in range(1, steps):
            print('--------------------------------------')
            print('LOAD STEP ', i)

            force += maxForce/steps
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(objective, Uu, p, self.trSettings, solver_algorithm=solver)
            
            self.write_output(Uu, p, i)
            self.disp = onp.array(np.append(self.disp,self.dispmax))
            self.force = onp.array(np.append(self.force,self.forcev))
        
        dispf = onp.append(self.disp,self.force,axis=0)    

        T_matrix_file_2 = h5p.File('./FDPlot_Buckling_Arch_BF_C.h5','w')
        T_matrix_file_2.create_dataset('FD', data=dispf)
        T_matrix_file_2.close()

        unload = False
        if unload:
            for i in range(steps, 2*steps - 1):
                print('--------------------------------------')
                print('LOAD STEP ', i)

                force -= maxForce/steps
                p = Objective.param_index_update(p, 0, force)
                Uu, solverSuccess = EqSolver.nonlinear_equation_solve(objective, Uu, p, self.trSettings, solver_algorithm=solver)
            
                self.write_output(Uu, p, i)
        
app = TractionArch()
app.setUp()
with Timer(name="AppRun"):
    app.run()
    
