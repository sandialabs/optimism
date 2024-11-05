import jax
import jax.numpy as np
import numpy as onp
from optimism import EquationSolver as EqSolver
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
        self.w = 0.25
        #h = 20
        #N = 5
        #M = 80
        #mesh, _ = self.create_mesh_and_disp(N, M, [-self.w/2,self.w/2], [0., h], lambda x: None)
        mesh = read__mesh('./foreground_mesh_optimism_buckle_D.exo')#Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, copyNodeSets=False)
        nodeSets = Mesh.create_nodesets_from_sidesets(mesh)
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)

        ebcs = [EssentialBC(nodeSet='iside_b0_0_b1_1', component=0),
                EssentialBC(nodeSet='iside_b0_0_b1_2', component=0),
                EssentialBC(nodeSet='iside_b0_0_b1_2', component=1)]

        self.dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

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

        self.compute_bc_reactions = jax.jit(jax.grad(self.energy_from_bcs, 1))

        self.trSettings = EqSolver.get_settings(max_trust_iters=400, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)

        self.outputForce = [0.0]
        self.outputDisp = [0.0]

    def energy_function_from_full_field(self, U, p):
        internalVariables = p[1]
        strainEnergy = self.bvpFuncs.compute_strain_energy(U, internalVariables)
        F = p[0]
        loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, self.lineQuadRule, self.mesh.sideSets['iside_b0_0_b1_1'],
                                                                    lambda x, n: np.array([0.0, -F/self.w]))
        return strainEnergy + loadPotential

    def energy_from_bcs(self, Uu, Ubc, p):
        U = self.dofManager.create_field(Uu, Ubc)
        return self.energy_function_from_full_field(U, p)

    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        return self.energy_function_from_full_field(U, p)

    def assemble_sparse(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        elementStiffnesses =  self.bvpFuncs.compute_element_stiffnesses(U, internalVariables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.fs.mesh.conns,
                                                                      self.dofManager)

    def write_output(self, Uu, p, step):
        U = self.create_field(Uu, p)
        plotName = 'euler_column_D-'+str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        Ubc = self.get_ubcs(p)
        internalVariables = p[1]
        rxnBc = self.compute_bc_reactions(Uu, Ubc, p)
        reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
        writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

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
        force2 = np.sum(reactions[:,1])
        print("applied force, reaction", force, force2)
        disp = np.max(np.abs(U[self.mesh.nodeSets['iside_b0_0_b1_1'],1]))
        self.outputForce.append(float(force))
        self.outputDisp.append(float(disp))
        print('Max Displacement')
        print(disp)

        with open('column_Fd.npz','wb') as f:
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
        objective = Objective.Objective(self.energy_function, Uu, p, 1.0,  precondStrategy)

        self.write_output(Uu, p, step=0)

        steps = 2
        maxForce = 0.0022
        for i in range(1, steps):
            key = jax.random.key(0)
            init_guess = Uu+jax.random.uniform(key=key,shape=np.shape(Uu)[0],minval=0,maxval=0.001)
            print('--------------------------------------')
            print('LOAD STEP ', i)
            force += maxForce/steps
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(objective, init_guess, p, self.trSettings, solver_algorithm=solver)

            self.write_output(Uu, p, i)

        unload = False
        if unload:
            for i in range(steps, 2*steps - 1):
                print('--------------------------------------')
                print('LOAD STEP ', i)
                force -= maxForce/steps
                p = Objective.param_index_update(p, 0, force)
                Uu, solverSuccess = EqSolver.nonlinear_equation_solve(objective, Uu, p, self.trSettings, solver_algorithm=solver)

                self.write_output(Uu, p, i)

app = EulerBeam()
app.setUp()
with Timer(name="AppRun"):
    app.run()

