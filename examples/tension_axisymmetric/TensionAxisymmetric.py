from functools import partial
import jax
from jax import numpy as np
from matplotlib import pyplot as plt

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism.material import J2Plastic
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

from properties import props

subProblemSettings = EqSolver.get_settings(max_trust_iters=500,
                                           tr_size=10.0,
                                           min_tr_size=1e-14,
                                           tol=1e-7,
                                           max_cumulative_cg_iters=150,
                                           use_preconditioned_inner_product_for_cg=True)


class AxisymmetricTension:

    def __init__(self):
        self.R0 = 0.5*6.35
        self.L0 = 30.0
        N = 4
        M = 15
        L0 = self.L0
        R0 = self.R0
        xRange = [0.0, R0]
        yRange = [0.0, L0]

        pOrder = 1
        mesh = Mesh.construct_structured_mesh(N, M, xRange, yRange, pOrder)
        nodeSets = {'axis': np.flatnonzero(mesh.coords[:,0] < xRange[0] + 1e-8),
                    'top': np.flatnonzero(mesh.coords[:,1] > yRange[1] - 1e-8),
                    'right': np.flatnonzero(mesh.coords[:,0] > xRange[1] - 1e-8),
                    'bot': np.flatnonzero(mesh.coords[:,1] < yRange[0] + 1e-8)}
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        
        # pOrder = 2
        # mesh = ReadExodusMesh.read_exodus_mesh('CylindricalSmoothBar_R_3_175mm_M2_b.g')
        # mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=pOrder, useBubbleElement=True, createNodeSetsFromSideSets=True)
        # self.mesh = Mesh.mesh_with_coords(mesh, mesh.coords*1e3)
        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*pOrder)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule, mode2D='axisymmetric')
        
        ebcs = [FunctionSpace.EssentialBC(nodeSet='axis', component=0),
                FunctionSpace.EssentialBC(nodeSet='top', component=1),
                FunctionSpace.EssentialBC(nodeSet='bot', component=1)]

        self.dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)

        
        materialModel = J2Plastic.create_material_model_functions(props)

        self.mechanicsFunctions =  Mechanics.create_mechanics_functions(self.fs,
                                                                        "axisymmetric",
                                                                        materialModel)
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
        EbcIndex = (self.mesh.nodeSets['top'],1)
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

        eqpsField = internalVariables[:,:,J2Plastic.EQPS]
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

        self.outputForce.append(float(np.sum(reactions[self.mesh.nodeSets['top'],1])))
        self.outputDisp.append(float(p[0]))

        with open('force-displacement.npz', 'wb') as f:
            np.savez(f,
                     displacement=np.array(self.outputDisp),
                     force=np.array(self.outputForce))

            
    def run(self):       
        
        Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        
        yDisp = 0.0
        state = self.mechanicsFunctions.compute_initial_state()
        p = Objective.Params(yDisp, state)

        precondStrategy = Objective.PrecondStrategy(self.assemble_sparse)
        objective = Objective.Objective(self.energy_function, Uu, p, precondStrategy)
                
        self.write_output(Uu, p, step=0)
        
        N = 50
        maxDisp = self.L0*0.25
        for i in range(1, N+1):
            print('\n-----------------------------------')
            print('LOAD STEP ', i, '\n')
            yDisp += maxDisp/N
            p = Objective.param_index_update(p, 0, yDisp)
            Uu = EqSolver.nonlinear_equation_solve(objective, Uu, p, subProblemSettings)
            
            internalVariables = self.mechanicsFunctions.\
                compute_updated_internal_variables(self.create_field(Uu, p), p[1])
            p = Objective.param_index_update(p, 1, internalVariables)
            
            self.write_output(Uu, p, i)

         
if __name__=='__main__':
    app = AxisymmetricTension()
    app.run()
 
    
