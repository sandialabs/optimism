from matplotlib import pyplot as plt

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import Interpolants
from optimism.material import Neohookean as MatModel
from optimism import Mechanics
from optimism import Mesh
from optimism.Mesh import EssentialBC
from optimism.Mesh import DofManager
from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import QuadratureRule
from optimism.Timer import Timer
from optimism import VTKWriter
from optimism.test.MeshFixture import MeshFixture           


class ContactArch(MeshFixture):

    def setUp(self):
        self.w = 0.07
        self.archRadius = 1.5
        self.ballRadius = self.archRadius/5.0
        self.initialBallLoc = self.archRadius + self.w + self.ballRadius
        N = 5
        M = 65
        
        mesh, _ = \
            self.create_arch_mesh_disp_and_edges(N, M,
                                                 self.w, self.archRadius, 0.5*self.w)
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, copyNodeSets=False)
        nodeSets = Mesh.create_nodesets_from_sidesets(mesh)
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        
        EBCs = [EssentialBC(nodeSet='left', field=0),
                EssentialBC(nodeSet='left', field=1),
                EssentialBC(nodeSet='right', field=0),
                EssentialBC(nodeSet='right', field=1),
                EssentialBC(nodeSet='push', field=1)]
        self.dofManager = DofManager(self.mesh, self.mesh.coords.shape, EBCs)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)

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

        def compute_energy_from_bcs(Uu, Ubc, internalVariables):
            U = self.dofManager.create_field(Uu, Ubc)
            return self.bvpFuncs.compute_strain_energy(U, internalVariables)
        
        self.compute_bc_reactions = jit(value_and_grad(compute_energy_from_bcs, 1))
        
        self.trSettings = EqSolver.get_settings()
        
        self.outputForce = []
        self.outputDisp = []
        self.outputEnergy = []


    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        return self.bvpFuncs.compute_strain_energy(U, internalVariables)

    
    def assemble_sparse(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        elementStiffnesses =  self.bvpFuncs.compute_element_stiffnesses(U, internalVariables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.fs.mesh.conns,
                                                                      self.dofManager)


    def write_output(self, Uu, p, step):
        U = self.create_field(Uu, p)
        plotName = 'arch_bc-'+str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        
        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        Ubc = self.get_ubcs(p)
        internalVariables = p[1]
        energy, rxnBc = self.compute_bc_reactions(Uu, Ubc, internalVariables)
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

        self.outputForce.append(float(-np.sum(reactions[self.mesh.nodeSets['push'],1])))
        self.outputDisp.append(float(-p[0]))
        self.outputEnergy.append(float(energy))

        with open('arch_bc_Fd.npz','wb') as f:
            np.savez(f, force=np.array(self.outputForce),
                     displacement=np.array(self.outputDisp),
                     energy=np.array(self.outputEnergy))

            
    def get_ubcs(self, p):
        yLoc = p[0]
        V = np.zeros(self.mesh.coords.shape)
        index = (self.mesh.nodeSets['push'],1)
        V = V.at[index].set(yLoc)
        return self.dofManager.get_bc_values(V)

    
    def create_field(self, Uu, p):
            return self.dofManager.create_field(Uu, self.get_ubcs(p))

        
    def run(self):
        Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        disp = 0.0
        ivs = self.bvpFuncs.compute_initial_state()
        p = Objective.Params(disp, ivs)

        precondStrategy = Objective.PrecondStrategy(self.assemble_sparse)
        objective = Objective.Objective(self.energy_function, Uu, p, precondStrategy)

        self.write_output(Uu, p, step=0)
        
        steps = 20
        maxDisp = 1.9*self.archRadius
        for i in range(1, steps):
            print('--------------------------------------')
            print('LOAD STEP ', i)

            disp -= maxDisp/steps
            p = Objective.param_index_update(p, 0, disp)
            Uu = EqSolver.nonlinear_equation_solve(objective,
                                                   Uu,
                                                   p,
                                                   self.trSettings)
            
            self.write_output(Uu, p, i)

        
app = ContactArch()
app.setUp()
with Timer(name="AppRun"):
    app.run()
    
    
