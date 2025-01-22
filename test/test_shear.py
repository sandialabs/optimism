# Test Case - Displacement based Shear Test
# ---------------------------------------------------
# Note - This approach involves segregation of master and slave DOFs
# ---------------------------------------------------

import sys
sys.path.insert(0, "/home/sarvesh/Documents/Github/optimism")


import jax.numpy as np
from jax import jit, grad

from optimism import EquationSolver as EqSolver
from optimism import QuadratureRule
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import FunctionSpace
from optimism.Mesh import create_higher_order_mesh_from_simplex_mesh
from optimism.material import Neohookean
from optimism.FunctionSpace import DofManagerMPC

from MeshFixture import *
import matplotlib.pyplot as plt



class ShearTest(MeshFixture):

    def setUp(self):
        dummyDispGrad = np.eye(2)
        self.mesh = self.create_mesh_disp_and_nodeset_layers(5, 5, [0.,1.], [0.,1.],
                                                             lambda x: dummyDispGrad.dot(x))[0]
        self.mesh = create_higher_order_mesh_from_simplex_mesh(self.mesh, order=1, createNodeSetsFromSideSets=True)
        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        self.EBCs = [FunctionSpace.EssentialBC(nodeSet='top', component=0),
                     FunctionSpace.EssentialBC(nodeSet='top', component=1),
                     FunctionSpace.EssentialBC(nodeSet='bottom', component=0),
                     FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        # dofManager = DofManagerMPC(self.fs, 2, self.EBCs, self.mesh)
        
        # self.master_array = dofManager.isMaster
        # self.slave_array = dofManager.isSlave
        # self.assembly = dofManager.dofAssembly
        # print("Assembly: ", self.assembly)

        shearModulus = 0.855 # MPa
        bulkModulus = 1000*shearModulus # MPa 
        youngModulus = 9.0*bulkModulus*shearModulus / (3.0*bulkModulus + shearModulus)
        poissonRatio = (3.0*bulkModulus - 2.0*shearModulus) / 2.0 / (3.0*bulkModulus + shearModulus)
        props = {
            'elastic modulus': youngModulus,
            'poisson ratio': poissonRatio,
            'version': 'coupled'
        }
        self.mat = Neohookean.create_material_model_functions(props)

        self.freq = 0.3 / 2.0 / np.pi 
        self.cycles = 1
        self.steps_per_cycle = 64
        self.steps = self.cycles*self.steps_per_cycle
        totalTime = self.cycles / self.freq
        self.dt = totalTime / self.steps
        self.maxDisp = 1.0

    def run(self):
        mechFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                         "plane strain",
                                                         self.mat)
        dofManager = DofManagerMPC(self.fs, 2, self.EBCs, self.mesh)

        def create_field(Uu, disp):
            def get_ubcs(disp):
                V = np.zeros(self.mesh.coords.shape)
                index = (self.mesh.nodeSets['top'], 0)
                V = V.at[index].set(disp)
                return dofManager.get_bc_values(V)

            return dofManager.create_field(Uu, get_ubcs(disp))
        
        def energy_function_all_dofs(U, p):
            internalVariables = p.state_data
            return mechFuncs.compute_strain_energy(U, internalVariables, self.dt)

        def compute_energy(Uu, p):
            U = create_field(Uu, p.bc_data)
            return energy_function_all_dofs(U, p)

        nodal_forces = jit(grad(energy_function_all_dofs, argnums=0))
        integrate_free_energy = jit(mechFuncs.compute_strain_energy)

        def write_output(Uu, p, step):
            from optimism import VTKWriter
            U = create_field(Uu, p.bc_data)
            plotName = "../test_results/disp_based_shear_test/"+'output-'+str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='displ', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

            energyDensities, stresses = mechFuncs.\
                compute_output_energy_densities_and_stresses(U, p.state_data, self.dt)
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, energyDensities)
            cellStresses = FunctionSpace.project_quadrature_field_to_element_field(self.fs, stresses)
            writer.add_cell_field(name='strain_energy_density',
                                  cellData=cellEnergyDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.add_cell_field(name='stress',
                                  cellData=cellStresses,
                                  fieldType=VTKWriter.VTKFieldType.TENSORS)
            writer.write()

        Uu = dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        U = create_field(Uu, p.bc_data)
        self.objective = Objective.ObjectiveMPC(compute_energy, Uu, p, dofManager)

        index = (self.mesh.nodeSets['top'], 0)

        time = 0.0
        times = []
        externalWorkStore = []
        incrementalPotentialStore = []
        forceHistory = []
        dispHistory = []
        for step in range(1, self.steps+1):
            force_prev = np.array(nodal_forces(U, p).at[index].get())
            applied_disp_prev = U.at[index].get()

            # disp = self.maxDisp * np.sin(2.0 * np.pi * self.freq * time)
            disp = self.maxDisp * self.freq * time
            print("Displacement in this load step: ", disp)

            p = Objective.param_index_update(p, 0, disp)
            Uu, solverSuccess = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            write_output(Uu, p, step)
            
            force = np.array(nodal_forces(U, p).at[index].get())
            applied_disp = U.at[index].get()
            externalWorkStore.append( 0.5*np.tensordot((force + force_prev),(applied_disp - applied_disp_prev), axes=1) )
            incrementalPotentialStore.append(integrate_free_energy(U, ivs, self.dt))

            forceHistory.append( np.sum(force) )
            dispHistory.append(disp)

            times.append(time)
            time += self.dt




app = ShearTest()
app.setUp()
app.run()