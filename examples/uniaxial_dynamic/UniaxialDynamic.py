from jax import numpy as np

from optimism import EquationSolver
from optimism import FunctionSpace
#from optimism.material import LinearElastic as Material
from optimism.material import Neohookean as Material
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

E = 10.0
nu = 0.0
rho = 1.0

solverSettings = EquationSolver.get_settings(max_cg_iters=50,
                                             max_trust_iters=500,
                                             min_tr_size=1e-13,
                                             tol=4e-12,
                                             use_incremental_objective=False)

class UniaxialDynamic:

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
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, pOrder)

        nodeSets = {'left': np.flatnonzero(mesh.coords[:,0] < xRange[0] + 1e-8),
                    'right': np.flatnonzero(mesh.coords[:,0] > xRange[1] - 1e-8),
                    'bottom': np.flatnonzero(mesh.coords[:,1] < yRange[0] + 1e-8)}
        self.mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)
        
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(pOrder-1))
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        self.fieldShape = self.mesh.coords.shape
        self.dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebcs)

        props = {'version': 'coupled',
                 'elastic modulus': E,
                 'poisson ratio': nu,
                 'density': rho}
        
        materialModel = Material.create_material_model_functions(props)
        newmarkParams = Mechanics.NewmarkParameters(gamma=0.5, beta=0.25)
        
        self.dynamicsFunctions =  Mechanics.create_dynamics_functions(self.fs,
                                                                      "plane strain",
                                                                      materialModel,
                                                                      newmarkParams)
        
        self.outputTime = []
        self.outputDisp = []
        self.outputKineticEnergy = []
        self.outputStrainEnergy = []


    def assemble_sparse(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        dt = p[4][0] - p[4][1]
        UuPre = p.dynamic_data
        UPre = self.create_field(UuPre, p)
        elementHessians = self.dynamicsFunctions.compute_element_hessians(U, UPre, internalVariables, dt)
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementHessians,
                                                                      self.mesh.conns,
                                                                      self.dofManager)

        
    def energy_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVariables = p[1]
        dt = p.time[0] - p.time[1]
        UuPre = p.dynamic_data
        UPre = self.create_field(UuPre, p)
        return self.dynamicsFunctions.compute_algorithmic_energy(U, UPre, internalVariables, dt)


    def create_field(self, Uu, p):
        return self.dofManager.create_field(Uu, self.get_ubcs(p))

    
    def get_ubcs(self, p):
        U = np.zeros(self.fieldShape)
        return self.dofManager.get_bc_values(U)

    
    def get_initial_conditions(self):
        U0 = np.zeros(self.fieldShape)
        Uu0 = self.dofManager.get_unknown_values(U0)

        v0 = 1.0
        V0 = np.zeros(self.fieldShape).at[:,0].set(v0/self.L*self.mesh.coords[:,0])
        Vu0 = self.dofManager.get_unknown_values(V0)
        return Uu0, Vu0

    
    def write_output(self, Uu, Vu, p, step):
        print('writing output')
        vtkFileName = 'uniaxial-' + str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=vtkFileName)

        U = self.create_field(Uu, p)
        V = self.create_field(Vu, p)
        internalVariables = p[1]
        
        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)
        writer.add_nodal_field(name='velocity', nodalData=V, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        # Ubc = self.dofManager.get_bc_values(U)
        # _,rxnBc = self.compute_reactions_from_bcs(Uu, Ubc, internalVariables)
        # reactions = np.zeros(U.shape).at[self.dofManager.isBc].set(rxnBc)
        # writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

        # eqpsField = internalVariables[:,:,J2Plastic.EQPS]
        # cellEqpsField = FunctionSpace.\
        #     project_quadrature_field_to_element_field(self.fs, eqpsField)
        # writer.add_cell_field(name='eqps', cellData=cellEqpsField, fieldType=VTKWriter.VTKFieldType.SCALARS)
       
        strainEnergyDensities, stresses = \
            self.dynamicsFunctions.\
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

        UEnd = np.average(U[self.mesh.nodeSets['right'],0])
        KE = self.dynamicsFunctions.compute_output_kinetic_energy(V)
        SE = self.dynamicsFunctions.compute_output_strain_energy(U, internalVariables)
        self.outputDisp.append(UEnd)
        self.outputTime.append(p.time[0])
        self.outputKineticEnergy.append(KE)
        self.outputStrainEnergy.append(SE)

        with open('history.npz','wb') as f:
            np.savez(f,
                     time=np.array(self.outputTime),
                     displacement=np.array(self.outputDisp),
                     kinetic_energy=np.array(self.outputKineticEnergy),
                     strain_energy=np.array(self.outputStrainEnergy))

    
    def run(self):       
        c = np.sqrt(E/(1-nu**2)/rho)
        exactPeriod = 4.0*self.L/c
        print('exact period = ', exactPeriod)
        dt = exactPeriod/8.0/2
        totalTime = 4*exactPeriod
        N = round(totalTime/dt)
        Uu, Vu = self.get_initial_conditions()
        internalVariables = self.dynamicsFunctions.compute_initial_state()
        tOld = -dt
        t = 0.0
        p = Objective.Params(None,
                             internalVariables,
                             None,
                             None,
                             np.array([t, tOld]),
                             Uu)

        precondStrategy = Objective.PrecondStrategy(self.assemble_sparse)
        objective = Objective.ScaledObjective(self.energy_function, Uu, p, precondStrategy)
        #objective = Objective.Objective(self.energy_function, Uu, p)

        # no initial forces, A = 0
        Au = np.zeros(Uu.shape)
        
        self.write_output(Uu, Vu, p, step=0)

        for i in range(1, N+1):
            print('LOAD STEP ', i, '------------------------\n')
            tOld = t
            t += dt
            print('  t = ', t, '\tdt = ', dt)
            p = Objective.param_index_update(p, 4, np.array([t, tOld]))
            UuPredicted, Vu = self.dynamicsFunctions.predict(Uu, Vu, Au, dt)
            #print('Predicted Ux = \n', self.create_field(UuPredicted,p)[:,0])
            #print('Predicted Vx = \n', self.create_field(Vu,p)[:,0])
            p = Objective.param_index_update(p, 5, UuPredicted)

            Uu = EquationSolver.nonlinear_equation_solve(objective,
                                                         Uu,
                                                         p,
                                                         solverSettings,
                                                         useWarmStart=False)

            UuCorrection = Uu - UuPredicted
            #print('Corrected Ux = \n', self.create_field(UuCorrection,p)[:,0])
            Vu, Au = self.dynamicsFunctions.correct(UuCorrection, Vu, Au, dt)
            
            internalVariables = self.dynamicsFunctions.\
                compute_updated_internal_variables(self.create_field(Uu, p), p[1])
            p = Objective.param_index_update(p, 1, internalVariables)
            
            self.write_output(Uu, Vu, p, i)

            #print(self.create_field(Uu,p)[:,0])
            #print(self.create_field(Au,p)[:,0])
            #print(self.create_field(Vu,p)[:,0])

        
if __name__=='__main__':
    app = UniaxialDynamic()
    app.run()
    
