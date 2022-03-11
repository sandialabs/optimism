from scipy.sparse import diags as sp_sparse_diags

from optimism.JaxConfig import *
from optimism import AlSolver
from optimism import BoundConstrainedSolver
from optimism import BoundConstrainedObjective
from optimism import EquationSolver as EqSolver
from optimism import EquationSolverSubspace as EqSolverSubspace
from optimism import FunctionSpace
from optimism import Interpolants
from optimism.phasefield import PhaseField
from optimism.phasefield import PhaseFieldLorentzPlastic as MatModel
#from optimism.phasefield import PhaseFieldThreshold as MatModel
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

from properties import props


subProblemSettings = EqSolver.get_settings(max_trust_iters=500,
                                           tr_size=10.0,
                                           min_tr_size=1e-14,
                                           tol=1e-5,
                                           max_cumulative_cg_iters=150,
                                           use_preconditioned_inner_product_for_cg=True)

alSettings = AlSolver.get_settings(target_constraint_decrease_factor=0.9,
                                   tol=4e-5,
                                   num_initial_low_order_iterations=3,
                                   use_second_order_update=False,
                                   penalty_scaling=1.1)

phaseStiffnessRelativeTolerance = 1.0/20.0


def compute_row_sum_gram_matrix(fs, dummyU, dummyInternalVars):
    func = lambda u, uGrad, q, x: u
    compute = grad(FunctionSpace.integrate_over_block, 1)
    return compute(fs, dummyU, dummyInternalVars, func, fs.mesh.blocks['block_1'])


class AxisymmetricTensionFracture:

    def __init__(self):
        self.R0 = 6.35
        self.L0 = 30.0
        N = 4
        M = 15
        L0 = self.L0
        R0 = self.R0
        xRange = [0.0, R0]
        yRange = [0.0, L0]

        # coords, conns = Mesh.create_structured_mesh_data(N, M, xRange, yRange)
        # blocks = {'block_0': np.arange(conns.shape[0])}
        # mesh = Mesh.construct_mesh_from_basic_data(coords, conns, blocks=blocks)
        # pOrder = 2
        # #master = Interpolants.make_master_tri_bubble_element(degree=pOrder)
        # master = Interpolants.make_master_tri_element(degree=pOrder)
        # master1d = Interpolants.make_master_line_element(degree=pOrder)
        # mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, master, master1d)

        #nodeSets = {'top': np.flatnonzero(mesh.coords[:,1] > yRange[1] - 1e-8),
        #            'right': np.flatnonzero(mesh.coords[:,0] > xRange[1] - 1e-8),
        #            'bottom': np.flatnonzero(mesh.coords[:,1] < yRange[0] + 1e-8)}
        #self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)

        pOrder = 2
        mesh = ReadExodusMesh.read_exodus_mesh('CylindricalSmoothBar_R_3_175mm_M2_b.g')
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=pOrder, useBubbleElement=True, createNodeSetsFromSideSets=True)
        self.mesh = Mesh.mesh_with_coords(mesh, mesh.coords*1e3)

        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*pOrder+1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule, mode2D='axisymmetric')

        self.lumpedMass = compute_row_sum_gram_matrix(self.fs, self.mesh.coords[:,0], np.zeros(self.fs.shapes.shape))
        vol = np.sum(self.lumpedMass)
        exact = np.sum(self.fs.vols.ravel())
        print("row sum volume = ", vol, "exact = ", exact)
        
        EBCs = [Mesh.EssentialBC(nodeSet='axis', field=0),
                Mesh.EssentialBC(nodeSet='bot', field=1),
                Mesh.EssentialBC(nodeSet='top', field=1),
                Mesh.EssentialBC(nodeSet='bot', field=2),
                Mesh.EssentialBC(nodeSet='top', field=2)]
                #Mesh.EssentialBC(nodeSet='axis', field=2)]

        self.fieldShape = (self.mesh.coords.shape[0], 3)
        self.dofManager = Mesh.DofManager(self.mesh, self.fieldShape, EBCs)

        materialModel = MatModel.create_material_model_functions(props)

        self.bvpFunctions =  PhaseField.create_phasefield_functions(self.fs,
                                                                    "axisymmetric",
                                                                    materialModel)

        UIdsFull = self.dofManager.dofToUnknown.reshape(self.fieldShape)
        self.phaseIds = UIdsFull[self.dofManager.isUnknown[:,2],2]
        
        self.outputForce = [0.0]
        self.outputDisp = [0.0]

    
    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        return self.bvpFunctions.compute_internal_energy(U, internalVariables)


    @partial(jit, static_argnums=0)
    @partial(value_and_grad, argnums=2)
    def compute_reactions_from_bcs(self, Uu, Ubc, internalVariables):
        U = self.dofManager.create_field(Uu, Ubc)
        return self.bvpFunctions.compute_internal_energy(U, internalVariables)


    def create_field(self, Uu, p):
        return self.dofManager.create_field(Uu, self.get_ubcs(p))

    
    def get_ubcs(self, p):
        endDisp = p[0]
        EbcIndex = ops.index[self.mesh.nodeSets['top'],1]
        V = ops.index_update(np.zeros(self.fieldShape), EbcIndex, endDisp)
        return self.dofManager.get_bc_values(V)

        
    def write_output(self, Uu, p, lam, step):
        vtkFileName = 'uniaxial-' + str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=vtkFileName)

        U = self.create_field(Uu, p)
        internalVariables = p[1]
        
        writer.add_nodal_field(name='displacement', nodalData=U[:,:2],
                               fieldType=VTKWriter.VTKFieldType.VECTORS)

        writer.add_nodal_field(name='phase', nodalData=U[:,2],
                               fieldType=VTKWriter.VTKFieldType.SCALARS)

        writer.add_nodal_field(name='phase_multiplier', nodalData=lam,
                               fieldType=VTKWriter.VTKFieldType.SCALARS)

        writer.add_nodal_field(name='nodal_mass', nodalData=self.lumpedMass,
                               fieldType=VTKWriter.VTKFieldType.SCALARS)
        
        writer.add_nodal_field(name='bcs',
                               nodalData=np.array(self.dofManager.isBc, dtype=int),
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)

        print("computing reactions")
        Ubc = self.dofManager.get_bc_values(U)
        _,rxnBc = self.compute_reactions_from_bcs(Uu, Ubc, internalVariables)
        reactions = ops.index_update(np.zeros(U.shape),
                                     ops.index[self.dofManager.isBc],
                                     rxnBc)
        writer.add_nodal_field(name='reactions',
                               nodalData=reactions,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)

        eqpsField = internalVariables[:,:,MatModel.STATE_EQPS]
        cellEqpsField = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, eqpsField)
        writer.add_cell_field(name='eqps',
                              cellData=cellEqpsField,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)

        print("computing stresses")
        lagrangianDensities, fluxes = \
            self.bvpFunctions.\
            compute_output_energy_densities_and_fluxes(U, internalVariables)
        stresses = fluxes[:,:,:3,:3]
        cellStresses = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, stresses)
        writer.add_cell_field(name='stress',
                              cellData=cellStresses,
                              fieldType=VTKWriter.VTKFieldType.TENSORS)

        strainEnergyDensities = self.bvpFunctions.compute_strain_energy_densities(U, internalVariables)
        cellStrainEnergyDensities = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, strainEnergyDensities)
        writer.add_cell_field(name='strain_energy',
                              cellData=cellStrainEnergyDensities,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)

        # voidField = internalVariables[:,:,MatModel.STATE_VOID_FRACTION]
        # cellVoidField = FunctionSpace.\
        #     project_quadrature_field_to_element_field(self.fs, voidField)
        # writer.add_cell_field(name='void_fraction',
        #                       cellData=cellVoidField,
        #                       fieldType=VTKWriter.VTKFieldType.SCALARS)

        # cellCritEnergyDensityField = (1.0 - cellVoidField)*props['critical strain energy density']
        # writer.add_cell_field(name='crit_strain_energy_density',
        #                       cellData=cellCritEnergyDensityField,
        #                       fieldType=VTKWriter.VTKFieldType.SCALARS)
        
        print("writing VTK file")
        writer.write()

        self.outputForce.append(float(np.sum(reactions[self.mesh.nodeSets['top'],1])))
        self.outputDisp.append(float(p[0]))

        with open('force-displacement.npz', 'wb') as f:
            np.savez(f, displacement=self.outputDisp, force=self.outputForce)


    def run(self):       
        
        Uu = self.dofManager.get_unknown_values(np.zeros(self.fieldShape))
        
        yDisp = 0.0
        internalVariables = self.bvpFunctions.compute_initial_state()
        p = Objective.Params(yDisp, internalVariables)

        def assemble_objective_stiffness(Uu, p, useBlockDiagonal=False):
            U = self.create_field(Uu, p)
            internalVars = p[1]

            if useBlockDiagonal:
                elementKMatrices = self.bvpFunctions.\
                    compute_block_diagonal_element_stiffnesses(U, internalVars)
            else:
                elementKMatrices = self.bvpFunctions.\
                    compute_element_stiffnesses(U, internalVars)

            return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementKMatrices,
                                                                          self.mesh.conns,
                                                                          self.dofManager)
        
        precondStrategy = Objective.TwoTryPrecondStrategy\
            (partial(assemble_objective_stiffness, useBlockDiagonal=False),
             partial(assemble_objective_stiffness, useBlockDiagonal=True))

        objective = BoundConstrainedObjective.BoundConstrainedObjective(self.energy_function,
                                                                        Uu,
                                                                        p,
                                                                        self.phaseIds,
                                                                        constraintStiffnessScaling=phaseStiffnessRelativeTolerance,
                                                                        precondStrategy=precondStrategy)
                
        self.write_output(Uu, p, objective.lam, step=0)
        
        N = 25
        maxDisp = self.L0*0.25
        for i in range(1, N+1):
            print('LOAD STEP ', i, '------------------------\n')
            yDisp += maxDisp/N
            p = Objective.param_index_update(p, 0, yDisp)
            
            Uu = BoundConstrainedSolver.bound_constrained_solve(objective,
                                                                Uu,
                                                                p,
                                                                alSettings,
                                                                subProblemSettings)
            #sub_problem_solver = EqSolverSubspace.trust_region_subspace_minimize

            Ubc = self.get_ubcs(p)
            U = self.dofManager.create_field(Uu, Ubc)
            internalVariables = self.bvpFunctions.\
                compute_updated_internal_variables(U, p[1])
            p = Objective.param_index_update(p, 1, internalVariables)
            objective.p = p

            print('writing output')
            self.write_output(Uu, p, objective.lam, i)

        
if __name__=='__main__':
    app = AxisymmetricTensionFracture()
    app.run()

