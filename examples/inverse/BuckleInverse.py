import jax
from jax import numpy as np

from optimism import ReadMesh

from optimism.material import Neohookean

from optimism import EquationSolver

from optimism.SparseCholesky import SparseCholesky as Cholesky
from optimism import SparseMatrixAssembler
from optimism import FunctionSpace
from optimism import Mechanics

from optimism import VTKWriter
from optimism import Mechanics
from optimism import QuadratureRule

from optimism.inverse import TopOpt
from optimism.inverse import NonlinearSolve
from optimism.FunctionSpace import EssentialBC, DofManager
from optimism import Objective
from jax.example_libraries import optimizers

props = {'elastic modulus': 10.0,
         'poisson ratio': 0.3}

materialModel = Neohookean.create_material_model_functions(props)

settings = EquationSolver.get_settings(max_cg_iters=50,
                                       max_trust_iters=200,
                                       min_tr_size=1e-13,
                                       tol=5e-11)

top = 'nodeset_1'
topSide = 'sideset_1'
bottom = 'nodeset_2'

class Buckle:

    def __init__(self, useTraction=True, meshWide=False, constrainXOnTop=True, useNewton=False):

        self.useTraction=useTraction
        self.meshWide=meshWide
        self.useNewton = useNewton

        if self.meshWide:
            meshName = 'buckleWide2d.json'
        else:
            meshName = 'buckleThin2d.json'
            
        self.mesh = ReadMesh.read_json_mesh(meshName)

        self.U = np.zeros(self.mesh.coords.shape)
        
        EBCs = [EssentialBC(nodeSet=bottom, component=0),
                EssentialBC(nodeSet=bottom, component=1)]
        
        if constrainXOnTop:
            EBCs.append(EssentialBC(nodeSet=top, component=0))

        if not self.useTraction:
            EBCs.append(EssentialBC(nodeSet=top, component=1))

        # We need a function space to create the DofManager, so we'll
        # make a dummy function space now that we don't use later. We will
        # construct a new function space in the energy function so that it is
        # sensitive to the node coordinate changes, which are the design parameters.
        # We can keep the same DofManager object between design iterations, since 
        # we never change which nodes have essential boundary conditions.
        self.triQuadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(self.mesh, self.triQuadRule)
        self.dofManager = DofManager(fs, self.mesh.coords.shape[1], EBCs)
        
        self.quadRule = QuadratureRule.create_quadrature_rule_1D(2)

        self.Uu = self.dofManager.get_unknown_values(self.U)


        
    def energy_func(self, Uu, p):
        designParams = p[2]        
        fs = FunctionSpace.construct_weighted_function_space(self.mesh,
                                                             self.triQuadRule,
                                                             quadratureWeights=designParams)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)
            
        return compute_energy(Uu, p,
                              fs, self.dofManager,
                              mechFuncs, self.quadRule, self.useTraction)


    def design_objective(self, Uu, p):
        mechanicalPenaltyFactor = 60.

        designParams = p[2]
        fs = FunctionSpace.construct_weighted_function_space(self.mesh,
                                                             self.triQuadRule,
                                                             quadratureWeights=designParams)

        #mechFuncs = Mechanics.create_mechanics_functions(fs,
        #                                                 'plane strain',
        #                                                 materialModel)

        U = create_field(Uu, p, self.mesh, self.dofManager, self.useTraction)
        mechanicalEnergy = -Mechanics.compute_traction_potential_energy(fs,
                                                                        U,
                                                                        self.quadRule,
                                                                        self.mesh.sideSets[topSide],
                                                                        lambda x, n: np.array([0.0, p[0][0]]))
        #mechanicalEnergy = -compute_energy(Uu, p,
        #                                   fs, self.dofManager,
        #                                   mechFuncs, self.quadRule, False)


        totalVolContrib = np.sum(fs.vols.ravel())

        return totalVolContrib + mechanicalPenaltyFactor * mechanicalEnergy


    def assemble_sparse(self, Uu, p):
        designParams = p[2]
        fs = FunctionSpace.construct_weighted_function_space(self.mesh,
                                                             self.triQuadRule,
                                                             quadratureWeights=designParams)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)

        U = create_field(Uu, p, self.mesh, self.dofManager, self.useTraction)
        elementStiffnesses = mechFuncs.compute_element_stiffnesses(U, p[1])
        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                      self.mesh.conns,
                                                                      self.dofManager)
        
        
    def plot_solution(self, U, plotName, p):        
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=U,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        
        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs',
                               nodalData=bcs,
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)

        elementPhases = p[2]
        fs = FunctionSpace.construct_weighted_function_space(self.mesh,
                                                             self.triQuadRule,
                                                             quadratureWeights=elementPhases)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)

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
        
        writer.add_cell_field(name='phase',
                              cellData=elementPhases,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)

        
        writer.write()


    def get_output_name(self, N):
        return 'euler-'+str(N).zfill(3)

        
    def run(self):

        chi = 11 * np.ones( (self.mesh.conns.shape[0], QuadratureRule.len(self.triQuadRule)) )
        elementPhases = TopOpt.compute_phases(chi)
        fs = FunctionSpace.construct_weighted_function_space(self.mesh,
                                                             self.triQuadRule,
                                                             quadratureWeights=elementPhases)
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         'plane strain',
                                                         materialModel)
        state = mechFuncs.compute_initial_state()

        p = Objective.Params( np.array([0.0,0.0]), state, elementPhases )
        self.objective = Objective.Objective(self.energy_func, self.Uu, p,
                                             Objective.PrecondStrategy(self.assemble_sparse))
        
        outputDisp = [0,]
        outputForce = [0,]

        reaction_func = jax.jit( lambda x,p: jax.grad(self.energy_func,1)(x,p)[0][1] )
        
        N = 1

        if self.useTraction: # this is traction for this case
            if self.meshWide:
                #maxDispOrTraction = -0.08
                maxDispOrTraction = -0.06
            else:
                maxDispOrTraction = -0.014
        else:
            maxDispOrTraction = -4.0

        for i in range(N):
            print("")
            print("----- LOAD STEP " + str(i+1) + " -----\n")

            endDispOrTraction = maxDispOrTraction*(i+1)/N

            p = Objective.param_index_update(p, 0, np.array([endDispOrTraction, 0.0]))
            self.Uu = EquationSolver.nonlinear_equation_solve(self.objective, self.Uu, p, settings)

            # post process
            print('step ', i)
            
            self.U = create_field(self.Uu, p, self.mesh, self.dofManager, self.useTraction)
            if self.useTraction:
                outputForce.append(reaction_func(self.Uu, p))
                topIndex = (self.mesh.nodeSets[top],1)
                topDispField = self.U[topIndex]
                outputDisp.append(np.average(topDispField))
            else:
                outputDisp.append(endDispOrTraction)
                outputForce.append(reaction_func(self.Uu, p))

            with open('force_disp.npz', 'wb') as f:
                np.savez(f,
                         disp=np.array(outputDisp),
                         force=np.array(outputForce))

            self.plot_solution(self.U, self.get_output_name(i), p)

            
        def loss(chi):
            elementPhases = TopOpt.compute_phases(chi)
            Uu = NonlinearSolve.nonlinear_solve(self.objective, settings, self.Uu, elementPhases)
            q = Objective.param_index_update(self.objective.p, 2, elementPhases)
            return self.design_objective(Uu, q)
        

        learningRate = 0.2
        numLearningSteps = 200
        
        opt_init, opt_update, get_params = optimizers.adam(learningRate)
        opt_state = opt_init(chi)

        def step(step, opt_state):
            value, grads = jax.value_and_grad(loss)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state
        
        for i in range(numLearningSteps):
            value, opt_state = step(i, opt_state)
            print('design objective = ', value)
            
            if value != value: return
            
            chi = get_params(opt_state)
            elementPhases = TopOpt.compute_phases(chi)
            q = Objective.param_index_update(self.objective.p, 2, elementPhases)

            self.Uu = NonlinearSolve.nonlinear_solve(self.objective, settings, self.Uu, elementPhases)
            
            self.U = create_field(self.Uu, q, self.mesh, self.dofManager, self.useTraction)
            self.plot_solution(self.U, self.get_output_name(i+1), q)
            
            
def create_field(Uu, p, mesh, dofManager, useTraction):
    return dofManager.create_field(Uu, get_ubcs(p, mesh, dofManager, useTraction))


def get_ubcs(p, mesh, dofManager, useTraction):
    V = np.zeros(dofManager.ids.shape)
    if not useTraction:
        yTopIndices = mesh.nodeSets[top][1]
        V = V.at[yTopIndices].set(p[0][0]),
        
    yBottomIndices = mesh.nodeSets[bottom][1]
    V = V.at[yBottomIndices].set(p[0][1])
    
    return dofManager.get_bc_values(V)

                
def compute_energy(Uu, p, fs, dofManager, mechFuncs, edgeQuadRule, useTraction):
    U = create_field(Uu, p, fs.mesh, dofManager, useTraction)
    mechanicalEnergy = mechFuncs.compute_strain_energy(U, p[1])
    tractionEnergy = Mechanics.compute_traction_potential_energy(fs, U, edgeQuadRule, fs.mesh.sideSets[topSide], 
                                                                  lambda x, n: np.array([0.0, p[0][0]])) if useTraction else 0.
    
    return mechanicalEnergy + tractionEnergy


from shutil import copyfile


app = Buckle(useTraction=True, meshWide=True, constrainXOnTop=True, useNewton=False)
app.run()


if False:

    app = Buckle(useTraction=True, meshWide=False, constrainXOnTop=True, useNewton=False)
    app.run()
    copyfile('force_disp.npz', 'thin_traction/force_disp.npz')
    
    app = Buckle(useTraction=True, meshWide=False, constrainXOnTop=True, useNewton=True)
    app.run()
    copyfile('force_disp.npz', 'newton_thin_traction/force_disp.npz')

    app = Buckle(useTraction=True, meshWide=True, constrainXOnTop=True, useNewton=False)
    app.run()
    copyfile('force_disp.npz', 'wide_traction/force_disp.npz')
    
    app = Buckle(useTraction=True, meshWide=True, constrainXOnTop=True, useNewton=True)
    app.run()
    copyfile('force_disp.npz', 'newton_wide_traction/force_disp.npz')
    
    app = Buckle(useTraction=False, meshWide=False, constrainXOnTop=True, useNewton=False)
    app.run()
    copyfile('force_disp.npz', 'thin/force_disp.npz')

    app = Buckle(useTraction=False, meshWide=False, constrainXOnTop=True, useNewton=True)
    app.run()
    copyfile('force_disp.npz', 'newton_thin/force_disp.npz')

    app = Buckle(useTraction=False, meshWide=True, constrainXOnTop=True, useNewton=False)
    app.run()
    copyfile('force_disp.npz', 'wide/force_disp.npz')

    app = Buckle(useTraction=False, meshWide=True, constrainXOnTop=True, useNewton=True)
    app.run()
    copyfile('force_disp.npz', 'newton_wide/force_disp.npz')



