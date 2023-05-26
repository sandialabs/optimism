import jax
from jax import numpy as np

from optimism.material import Neohookean
from optimism.material import LinearElastic
from optimism import EquationSolver

from optimism import SparseMatrixAssembler
from optimism import FunctionSpace
from optimism import Mechanics

from optimism import VTKWriter
from optimism import Mechanics
from optimism import QuadratureRule

from optimism.inverse import NonlinearSolve
from optimism.FunctionSpace import EssentialBC, DofManager
from optimism.Timer import Timer
from optimism import Objective
from jax.example_libraries import optimizers
from optimism.test.MeshFixture import MeshFixture

shapeOpt = True

if shapeOpt:
    import optimism.inverse.ShapeOpt as DesignOpt
else:
    import optimism.inverse.TopOpt as DesignOpt

    
props = {'elastic modulus': 10.0,
         'poisson ratio': 0.3}

dProps = {'elastic modulus': 1e-1,
          'poisson ratio': 0.0}

materialModel = Neohookean.create_material_model_functions(props)
meshDistortionModel = Neohookean.create_material_model_functions(dProps)

settings = EquationSolver.get_settings(max_cg_iters=100,
                                       max_trust_iters=500,
                                       min_tr_size=1e-13,
                                       tol=1e-10)

maxTraction = 0.025
numSteps = 15

N = 8
M = 56

top = 'nodeset_1'
topSide = 'sideset_1'
bottom = 'nodeset_2'


class Buckle(MeshFixture):

    def __init__(self):

        self.designStep=0
        self.w = 0.07
        self.archRadius = 1.5
        
        self.mesh, _ = \
            self.create_arch_mesh_disp_and_edges(N, M,
                                                 self.w, self.archRadius, 0.5*self.w)

        assert(self.mesh.nodeSets['push'].shape[0] == 2)
        
        EBCs = [EssentialBC(nodeSet='left', component=0),
                EssentialBC(nodeSet='left', component=1),
                EssentialBC(nodeSet='right', component=0),
                EssentialBC(nodeSet='right', component=1)]
        
        # We need a function space to create the DofManager, so we'll
        # make a dummy function space now that we don't use later. We will
        # construct a new function space in the energy function so that it is
        # sensitive to the node coordinate changes, which are the design parameters.
        # We can keep the same DofManager object between design iterations, since 
        # we never change which nodes have essential boundary conditions.
        self.triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(self.mesh, self.triQuadRule)
        self.dofManager = DofManager(fs, self.mesh.coords.shape[1], EBCs)

        self.U = np.zeros(self.mesh.coords.shape)
        self.Uu = self.dofManager.get_unknown_values(self.U)

        meshfs = FunctionSpace.construct_weighted_function_space(self.mesh, self.triQuadRule)
        self.meshMechFuncs = Mechanics.create_mechanics_functions(meshfs,
                                                                  'plane strain',
                                                                  meshDistortionModel)
        self.meshMechState = self.meshMechFuncs.compute_initial_state()
        
    def energy_func(self, Uu, p):
        fs = DesignOpt.create_function_space(self.mesh,
                                             self.triQuadRule,
                                             p)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)
        U = create_field(Uu, p, self.mesh, self.dofManager)
        pushDisps = U[self.mesh.nodeSets['push'],1]
        tractionEnergy = -p[0] * np.sum(pushDisps)
        mechanicalEnergy = mechFuncs.compute_strain_energy(U, p[1])
               
        return mechanicalEnergy + tractionEnergy


    def reaction_func(self, Uu, p):
        return jax.grad(self.energy_func,1)(Uu,p)[0]


    def compute_volume(self, p):
        fs = DesignOpt.create_function_space(self.mesh, self.triQuadRule, p)
        totalVolContrib = np.sum(fs.vols.ravel())
        return totalVolContrib


    def compute_mesh_mechanical_energy(self, p):
        return self.meshMechFuncs.compute_strain_energy(p[2]-self.mesh.coords, self.meshMechState) if shapeOpt else 0.0
        
    
    def assemble_sparse(self, Uu, p):
        fs = DesignOpt.create_function_space(self.mesh, self.triQuadRule, p)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)
        U = create_field(Uu, p, self.mesh, self.dofManager)
        elementStiffnesses = mechFuncs.compute_element_stiffnesses(U, p[1])
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.mesh.conns,
                                                                      self.dofManager)
    
    
    def create_params_from_design_vars(self, chi):
        return DesignOpt.create_parameters_from_design_vars(chi, self.mesh, self.dofManager)

    
    def simulate(self, designParams, output_func):
        traction = 0.0
        self.objective.p = Objective.param_index_update(self.objective.p, 0, traction)

        Uu = 0.0*self.Uu
        U = create_field(Uu, self.objective.p, self.mesh, self.dofManager)
        pushDisps = U[self.mesh.nodeSets['push'],1]

        output_func(U, self.objective.p, pushDisps, 0)
        
        work = 0.0
        for i in range(numSteps):
            tractionInc = maxTraction/numSteps
            traction -= tractionInc
            p = Objective.param_index_update(self.objective.p, 0, traction)
            Uu = NonlinearSolve.nonlinear_solve(self.objective, settings, Uu, designParams)
            
            U = create_field(Uu, p, self.mesh, self.dofManager)
            pushDispsNew = U[self.mesh.nodeSets['push'],1]
            work += p[0] * np.sum(pushDispsNew - pushDisps)
            pushDisps = pushDispsNew

            output_func(U, p, pushDisps, i+1)

        # reverse the loading
        for i in range(numSteps):
            tractionInc = maxTraction/numSteps
            traction += tractionInc
            p = Objective.param_index_update(self.objective.p, 0, traction)
            Uu = NonlinearSolve.nonlinear_solve(self.objective, settings, Uu, designParams)
            
            U = create_field(Uu, p, self.mesh, self.dofManager)
            pushDispsNew = U[self.mesh.nodeSets['push'],1]
            work += p[0] * np.sum(pushDispsNew - pushDisps)
            pushDisps = pushDispsNew

            output_func(U, p, pushDisps, numSteps+i+1)
            

        q = Objective.param_index_update(self.objective.p, 2, designParams)
        volume = self.compute_volume(q)
        meshMechanicalEnergy = 100*self.compute_mesh_mechanical_energy(q)
        work *= -100
        print('volume, work, mesh energy = ', volume, work, meshMechanicalEnergy)

        designObjective = work + volume + meshMechanicalEnergy
        if designObjective != designObjective:
            designObjective = np.inf
        return designObjective

    
    def run(self):
        chi = DesignOpt.create_initial_design_vars(self.mesh, self.dofManager, self.triQuadRule)
        designParams = DesignOpt.create_parameters_from_design_vars(chi, self.mesh, self.dofManager)

        initialDisp = 0.0
        p = Objective.Params(initialDisp, None, designParams)
        fs = DesignOpt.create_function_space(self.mesh, self.triQuadRule, designParams)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)
        state = mechFuncs.compute_initial_state()
        p = Objective.param_index_update(p, 1, state)
        self.objective = Objective.Objective(self.energy_func, self.Uu, p,
                                             Objective.PrecondStrategy(self.assemble_sparse))
        
        def loss(chi):
            print('forward run\n')
            designParams = self.create_params_from_design_vars(chi)
            return self.simulate(designParams, lambda U, p, disp, i: None)

        def debug_loss(chi):
            print('debug run\n')
            fs = DesignOpt.create_function_space(self.mesh, self.triQuadRule, self.objective.p[3])
            mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                             'plane strain',
                                                             materialModel)
            currentMesh = DesignOpt.create_current_mesh(self.mesh, self.objective.p[3])
            post_process = PostProcess(currentMesh, self.dofManager, self.triQuadRule, mechFuncs,
                                       'arch-'+str(self.designStep).zfill(2))
            self.designStep+=1
            designParams = self.create_params_from_design_vars(chi)
            return self.simulate(designParams, post_process)
        
               
        learningRate = 0.001
        numLearningSteps = 201
        
        opt_init, opt_update, get_params = optimizers.adam(learningRate)
        opt_state = opt_init(chi)

        def step(step, opt_state):
            value, grads = jax.value_and_grad(loss)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state
        
        for i in range(numLearningSteps):
            chi = get_params(opt_state)
            # set the parameters to the correct design variable outside of the A.D.
            # need to make sure this parameter is exposed internally to get correct gradients
            self.objective.p = Objective.param_index_update(self.objective.p, 2, self.create_params_from_design_vars(chi))
            
            if i%5==0:
                debug_loss(get_params(opt_state))
                
            value, opt_state = step(i, opt_state)
            print('design objective = ', value)
            if value != value: return


def create_field(Uu, p, mesh, dofManager):
    return dofManager.create_field(Uu, get_ubcs(p, mesh, dofManager))


def get_ubcs(p, mesh, dofManager):
    V = np.zeros(mesh.coords.shape)
    return dofManager.get_bc_values(V)


class PostProcess:

    def __init__(self, mesh, dofManager, triQuadRule, mechFuncs, filename):
        self.disp = [0,]
        self.force = [0,]

        self.mesh = mesh
        self.dofManager = dofManager
        self.triQuadRule = triQuadRule
        self.mechFuncs = mechFuncs
        self.filename=filename
        
    def __call__(self, U, p, pushDisps, i):
        self.disp.append(np.average(pushDisps))
        self.force.append(2*p[0])
        
        with open(self.filename+'.npz', 'wb') as f:
            np.savez(f,
                     disp=np.array(self.disp),
                     force=np.array(self.force))
            
        self.plot_solution(U, self.get_output_name(i), p)
            
    def get_output_name(self, N):
        return self.filename + '.' + str(N).zfill(3)

    def plot_solution(self, U, plotName, p):
        fs = DesignOpt.create_function_space(self.mesh, self.triQuadRule, p)

        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)
        
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=U,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        
        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs',
                               nodalData=bcs,
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)

        strainEnergyDensities, stresses = \
            mechFuncs.compute_output_energy_densities_and_stresses(U, p[1])

        strainEnergyDensities = np.squeeze(strainEnergyDensities)
        stresses = np.squeeze(stresses)

        writer.add_cell_field(name='strain_energy_density',
                              cellData=strainEnergyDensities,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)
        
        writer.add_cell_field(name='stress',
                              cellData=stresses,
                              fieldType=VTKWriter.VTKFieldType.TENSORS)

        if not shapeOpt:
            elementPhases = p[2]
            writer.add_cell_field(name='phase',
                                  cellData=elementPhases,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)

        writer.write()
        
        
##### RUN IT #####


app = Buckle()
app.run()
