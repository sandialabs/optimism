from functools import partial
import jax
from jax import numpy as np
from matplotlib import pyplot as plt

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism.material import J2Plastic as Material
#from optimism.material import Neohookean
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import VTKWriter


class Uniaxial:

    def __init__(self):
        self.w = 0.1
        self.L = 1.0
        N = 15
        M = 4
        L = self.L
        w = self.w
        xRange = [0.0, L]
        yRange = [0.0, w]

        coords, conns = Mesh.create_structured_mesh_data(N, M, xRange, yRange)
        blocks = {'block_0': np.arange(conns.shape[0])}
        mesh = Mesh.construct_mesh_from_basic_data(coords, conns, blocks=blocks)
        pOrder = 2
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=pOrder)

        nodeSets = {'left': np.flatnonzero(mesh.coords[:,0] < xRange[0] + 1e-8),
                    'right': np.flatnonzero(mesh.coords[:,0] > xRange[1] - 1e-8),
                    'bottom': np.flatnonzero(mesh.coords[:,1] < yRange[0] + 1e-8)}
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(pOrder-1))
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='right', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        self.dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)

        E = 10.0
        nu = 0.25
        Y0 = 0.01*E
        H = E/100
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'yield strength': Y0,
                 'hardening model': 'linear',
                 'hardening modulus': H}
        
        materialModel = Material.create_material_model_functions(props)

        self.mechanicsFunctions =  Mechanics.create_mechanics_functions(
            self.fs, "plane strain", materialModel)
        
        self.outputForce = []
        self.outputDisp = []


    def assemble_sparse(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        elementStiffnesses = self.mechanicsFunctions.\
            compute_element_stiffnesses(U, internalVariables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.mesh.conns,
                                                                      self.dofManager)

        
    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        return self.mechanicsFunctions.compute_strain_energy(U, internalVariables)


    @partial(jax.jit, static_argnums=0)
    @partial(jax.value_and_grad, argnums=2)
    def compute_reactions_from_bcs(self, Uu, Ubc, internalVariables):
        U = self.dofManager.create_field(Uu, Ubc)
        return self.mechanicsFunctions.compute_strain_energy(U, internalVariables)


    def create_field(self, Uu, p):
        return self.dofManager.create_field(Uu, self.get_ubcs(p))

    
    def get_ubcs(self, p):
        endDisp = p[0]
        EbcIndex = (self.mesh.nodeSets['right'],0)
        V = np.zeros_like(self.mesh.coords).at[EbcIndex].set(endDisp)
        return self.dofManager.get_bc_values(V)

        
    def write_output(self, Uu, p, step):
        print('writing output')
        vtkFileName = 'uniaxial-' + str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=vtkFileName)

        U = self.create_field(Uu, p)
        internalVariables = p[1]
        
        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        Ubc = self.dofManager.get_bc_values(U)
        _,rxnBc = self.compute_reactions_from_bcs(Uu, Ubc, internalVariables)
        reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
        writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

        if hasattr(Material, 'EQPS'):
            eqpsField = internalVariables[:,:,Material.EQPS]
            cellEqpsField = FunctionSpace.\
                project_quadrature_field_to_element_field(self.fs, eqpsField)
            writer.add_cell_field(name='eqps', cellData=cellEqpsField, fieldType=VTKWriter.VTKFieldType.SCALARS)
       
        strainEnergyDensities, stresses = \
            self.mechanicsFunctions.\
            compute_output_energy_densities_and_stresses(U, internalVariables)
        cellStrainEnergyDensities = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, strainEnergyDensities)
        cellStresses = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, stresses)
        writer.add_cell_field(name='strain_energy_density',
                              cellData=cellStrainEnergyDensities,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)
        writer.add_cell_field(name='stress',
                              cellData=cellStresses,
                              fieldType=VTKWriter.VTKFieldType.TENSORS)

        writer.write()

        self.outputForce.append(float(np.sum(reactions[self.mesh.nodeSets['right'],0])))
        self.outputDisp.append(float(p[0]))

        
    def run(self):       
        
        Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        settings = EqSolver.get_settings(max_cumulative_cg_iters=100,
                                         max_trust_iters=1000,
                                         use_preconditioned_inner_product_for_cg=True)
        
        xDisp = 0.0
        state = self.mechanicsFunctions.compute_initial_state()
        p = Objective.Params(xDisp, state)

        precondStrategy = Objective.PrecondStrategy(self.assemble_sparse)
        objective = Objective.ScaledObjective(self.energy_function, Uu, p, precondStrategy)
                
        self.write_output(Uu, p, step=0)
        
        N = 10
        maxDisp = self.L*0.05
        for i in range(1, N+1):
            print('LOAD STEP ', i, '------------------------\n')
            xDisp = i/N*maxDisp
            p = Objective.param_index_update(p, 0, xDisp)
            
            Uu = EqSolver.nonlinear_equation_solve(objective, Uu, p, settings)
            
            state = self.mechanicsFunctions.\
                compute_updated_internal_variables(self.create_field(Uu, p), p[1])
            p = Objective.param_index_update(p, 1, state)
            
            self.write_output(Uu, p, i)

            
    def make_FD_plot(self):
        plt.plot(self.outputDisp, self.outputForce, marker='o')
        plt.xlabel('Displacement')
        plt.ylabel('Force')
        plt.savefig('uniaxial_FD.pdf')

        
if __name__=='__main__':
    app = Uniaxial()
    app.run()
    app.make_FD_plot()
    
