import jax
from jax import numpy as np

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import Interpolants
#from optimism.material import Neohookean as MatModel
from optimism.material import J2Plastic as MatModel
from optimism import Mechanics
from optimism import Mesh
from optimism.FunctionSpace import EssentialBC, DofManager
from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import Surface
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import TractionBC
from optimism import VTKWriter

        
class TractionArch():

    def __init__(self):

        mesh = ReadExodusMesh.read_exodus_mesh('hemisphere_axisym.g')
        
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh,
                                                                    order=2,
                                                                    createNodeSetsFromSideSets=True)
        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=3)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        
        ebcs = [EssentialBC(nodeSet='axis', component=0),
                EssentialBC(nodeSet='rim', component=0),
                EssentialBC(nodeSet='rim', component=1),
                EssentialBC(nodeSet='push', component=1)]
        self.dofManager = DofManager(self.fs, self.mesh.coords.shape[1], ebcs)

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'yield strength': 0.5e3,
                 'hardening model': 'linear',
                 'hardening modulus': E/25}
        materialModel = MatModel.create_material_model_functions(props)

        self.bvpFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                             mode2D="axisymmetric",
                                                             materialModel=materialModel)

        def compute_energy_from_bcs(Uu, Ubc, p):
            U = self.dofManager.create_field(Uu, Ubc)
            internalVariables = p[1]
            strainEnergy = self.bvpFuncs.compute_strain_energy(U, internalVariables)
            return strainEnergy
        
        self.compute_bc_reactions = jax.jit(jax.grad(compute_energy_from_bcs, 1))
        
        self.trSettings = EqSolver.get_settings(max_trust_iters=400, t1=0.4, t2=1.5, eta1=1e-8, eta2=0.2, eta3=0.8, over_iters=100)
        
        self.outputForce = []
        self.outputDisp = []


    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        strainEnergy = self.bvpFuncs.compute_strain_energy(U, internalVariables)
        return strainEnergy

    
    def assemble_sparse(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        elementStiffnesses =  self.bvpFuncs.compute_element_stiffnesses(U, internalVariables)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.fs.mesh.conns,
                                                                      self.dofManager)


    def write_output(self, Uu, p, step):
        U = self.create_field(Uu, p)
        plotName = 'hemi_disp_control-'+str(step).zfill(3)
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

        if hasattr(MatModel, 'EQPS'):
            eqpsField = internalVariables[:,:,MatModel.EQPS]
            cellEqpsField = FunctionSpace.project_quadrature_field_to_element_field(self.fs, eqpsField)
            writer.add_cell_field(name='equiv_plastic_strain',
                                  cellData=cellEqpsField,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
        
        writer.write()

        disp = -p[0]
        force = -np.sum(reactions[self.mesh.nodeSets['push'],1])
        self.outputForce.append(float(force))
        self.outputDisp.append(float(disp))

        with open('disp_control_response.npz','wb') as f:
            np.savez(f, force=np.array(self.outputForce), displacement=np.array(self.outputDisp))

            
    def get_ubcs(self, p):
        V = np.zeros(self.mesh.coords.shape)
        uApplied = p[0]
        V = V.at[self.mesh.nodeSets['push'],1].set(uApplied)
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
        
        steps = 40
        maxDisp = -1.63
        for i in range(1, steps):
            print('--------------------------------------')
            print('LOAD STEP ', i)

            disp += maxDisp/steps
            p = Objective.param_index_update(p, 0, disp)
            Uu = EqSolver.nonlinear_equation_solve(objective, Uu, p, self.trSettings)

            
            U = self.dofManager.create_field(Uu, self.get_ubcs(p))
            ivs = self.bvpFuncs.\
                compute_updated_internal_variables(U, p[1])
            p = Objective.param_index_update(p, 1, ivs)
            objective.p = p
            
            self.write_output(Uu, p, i)

        unload = False
        if unload:
            for i in range(steps, 2*steps - 1):
                print('--------------------------------------')
                print('LOAD STEP ', i)

                disp -= maxDisp/steps
                p = Objective.param_index_update(p, 0, disp)
                Uu = EqSolver.nonlinear_equation_solve(objective, Uu, p, self.trSettings)

                U = self.dofManager.create_field(Uu, self.get_ubcs(p))
                ivs = self.bvpFuncs.\
                    compute_updated_internal_variables(U, p[1])
                p = Objective.param_index_update(p, 1, ivs)
                objective.p = p
                
                self.write_output(Uu, p, i)


if __name__ == '__main__':
    app = TractionArch()
    app.run()
    
