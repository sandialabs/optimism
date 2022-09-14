from functools import partial
import jax
from jax import numpy as np

from optimism import AlSolver
from optimism import BoundConstrainedSolver
from optimism import BoundConstrainedObjective
from optimism.ConstrainedObjective import ConstrainedObjective
from optimism import EquationSolver as EqSolver
from optimism import Mesh
from optimism import FunctionSpace
from optimism.phasefield import PhaseField
import optimism.phasefield.PhaseFieldThreshold as Model
from optimism import Objective
from optimism.SparseCholesky import SparseCholesky as Cholesky
from optimism import SparseMatrixAssembler
from optimism import QuadratureRule
from optimism import VTKWriter

E = 40.0
nu = 0.25
Gc = 1.0
ell = 0.1

props = {'elastic modulus': E,
         'poisson ratio': nu,
         'critical energy release rate': Gc,
         'regularization length': ell,
         'kinematics': 'large deformations'}

phaseStiffnessRelativeTolerance = 0.05


subProblemSettings = EqSolver.get_settings(max_trust_iters=500,
                                           tr_size=0.05,
                                           tol=1e-9,
                                           min_tr_size=1e-12,
                                           use_preconditioned_inner_product_for_cg=True)


alSettings = AlSolver.get_settings(target_constraint_decrease_factor=0.9,
                                   num_initial_low_order_iterations=2,
                                   tol=5e-9,
                                   max_gmres_iters=20)

class SharpNotchProblem:

    def __init__(self):
        self.Nx = 25
        self.Ny = 16
        xRange = [0.,1.]
        yRange = [0.,0.5]
        precrack = 0.25

        mesh = Mesh.construct_structured_mesh(self.Nx, self.Ny, xRange, yRange, elementOrder=1)
        nodeSets = {'bottom_symmetry': np.flatnonzero( (mesh.coords[:,1] < yRange[0] + 1e-8) &
                                                       (mesh.coords[:,0] > precrack - 1e-8) ),
                    'top': np.flatnonzero(mesh.coords[:,1] > yRange[1] - 1e-8)}
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        
        ebcs = [FunctionSpace.EssentialBC(nodeSet='top', component=0),
                FunctionSpace.EssentialBC(nodeSet='top', component=1),
                FunctionSpace.EssentialBC(nodeSet='bottom_symmetry', component=1)]
       
        dim = 3 # 3 = 2 displacement components + 1 scalar phase
        self.dofManager = FunctionSpace.DofManager(self.fs, dim, EssentialBCs=ebcs)

        materialModel = Model.create_material_model_functions(props)

        self.bvpFunctions = PhaseField.create_phasefield_functions(self.fs,
                                                                   "plane strain",
                                                                   materialModel)
        self.fieldShape = Mesh.num_nodes(self.mesh), dim
        UIdsFull = self.dofManager.dofToUnknown.reshape(self.fieldShape)
        self.phaseIds = UIdsFull[self.dofManager.isUnknown[:,2],2]
        
        self.outputForce = []
        self.outputDisp = []

        def get_ubcs(yDisp):
            V = np.zeros(self.fieldShape)
            index = (self.mesh.nodeSets['top'],1)
            V = V.at[index].set(yDisp)
            return self.dofManager.get_bc_values(V)

        self.get_ubcs = get_ubcs
        
        def energy(Uu, p):
            yDisp = p[0]
            Ubc = get_ubcs(yDisp)
            U = self.dofManager.create_field(Uu, Ubc)
            internalVariables = p[1]
            return self.bvpFunctions.compute_internal_energy(U, internalVariables)

        self.objective_function = energy

        def energy_for_rxns(Ubc, Uu, p):
            U = self.dofManager.create_field(Uu, Ubc)
            internalVariables = p[1]
            return self.bvpFunctions.compute_internal_energy(U, internalVariables)
        
        self.compute_reactions = jax.jit(jax.grad(energy_for_rxns))
    

    def plot_solution(self, U, p, lagrange, plotName):
        yDisp = p[0]
        internalVariables = p[1]
        
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
    
        writer.add_nodal_field(name='displacement', nodalData=U[:,:2],
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
    
        writer.add_nodal_field(name='phase', nodalData=U[:,2],
                               fieldType=VTKWriter.VTKFieldType.SCALARS)
    
        writer.add_nodal_field(name='bcs',
                               nodalData=np.array(self.dofManager.isBc, dtype=int),
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)

        rxnBc = self.compute_reactions(self.dofManager.get_bc_values(U),
                                       self.dofManager.get_unknown_values(U),
                                       p)
        reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
        writer.add_nodal_field(name='reactions',
                               nodalData=reactions,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)      

        writer.add_nodal_field(name='lagrange_multiplier',
                               nodalData=lagrange,
                               fieldType=VTKWriter.VTKFieldType.SCALARS)
        
        lagrangianDensities, fluxes = self.bvpFunctions.compute_output_energy_densities_and_fluxes(U, internalVariables)
        cellLDensities = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, lagrangianDensities)
        writer.add_cell_field(name='lagrangian_density',
                              cellData=cellLDensities,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)

        stresses = fluxes[:,:,:3,:3]
        cellStresses = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, stresses)
        writer.add_cell_field(name='stress',
                              cellData=cellStresses,
                              fieldType=VTKWriter.VTKFieldType.TENSORS)

        writer.write()

        self.outputForce.append(float(np.sum(reactions[self.mesh.nodeSets['top'],1])))
        self.outputDisp.append(float(yDisp))

        with open('force-displacement.npz', 'wb') as f:
            np.savez(f,
                     displacement=self.outputDisp,
                     force=self.outputForce)

            
    def assemble_sparse_hessian(self, Uu, p, useBlockDiagonal=False):
        yDisp = p[0]
        Ubc = self.get_ubcs(yDisp)
        U = self.dofManager.create_field(Uu, Ubc)
        internalVariables = p[1]
        if useBlockDiagonal:
            elementStiffnesses =  self.bvpFunctions.compute_block_diagonal_element_stiffnesses(U, internalVariables)
        else:
            elementStiffnesses =  self.bvpFunctions.compute_element_stiffnesses(U, internalVariables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.fs.mesh.conns,
                                                                      self.dofManager)

            
    def run(self):
        Uu = self.dofManager.get_unknown_values(np.zeros(self.fieldShape))

        yDisp = 0.0
        yDispInc = 0.02
        internalVariables = self.bvpFunctions.compute_initial_state()
        
        p = Objective.Params(yDisp, internalVariables)

        precondStrategy = Objective.TwoTryPrecondStrategy(partial(self.assemble_sparse_hessian, useBlockDiagonal=False),
                                                          partial(self.assemble_sparse_hessian, useBlockDiagonal=True))
              
        objective = BoundConstrainedObjective.BoundConstrainedObjective(
            self.objective_function, Uu, p, self.phaseIds,
            constraintStiffnessScaling=phaseStiffnessRelativeTolerance,
            precondStrategy=precondStrategy)

        lamField = np.zeros(self.fieldShape[0]).at[self.dofManager.isUnknown[:,2]].set(objective.get_multipliers())
        
        self.plot_solution(np.zeros(self.fieldShape), p, lamField, "sharpNotch-"+str(0).zfill(3))
        
        for step in range(1, 25):
            print('\n------------------------------------')
            print(" LOAD STEP: ", step)
            #print('initial energy', self.objective_function(Uu, p))
            #K = self.assemble_sparse_hessian(Uu, p)
            #with open('hessian.npy', 'wb') as file:
            #    np.save(file, K.todense())
            #print('saved initial hessian')
            yDisp += yDispInc
            
            p = Objective.param_index_update(p, 0, yDisp)
            
            Uu = BoundConstrainedSolver.bound_constrained_solve(objective,
                                                                Uu,
                                                                p,
                                                                alSettings,
                                                                subProblemSettings)

            Ubc = self.get_ubcs(yDisp)
            U = self.dofManager.create_field(Uu, Ubc)
            internalVariables = self.bvpFunctions.compute_updated_internal_variables(U, p[1])
            p = Objective.param_index_update(p, 1, internalVariables)
            objective.p = p

            lamField = np.zeros(self.fieldShape[0]).at[self.dofManager.isUnknown[:,2]].set(objective.get_multipliers())
            
            self.plot_solution(U, p, lamField, "sharpNotch-"+str(step).zfill(3))


if __name__ == '__main__':
    app = SharpNotchProblem()
    app.run()
