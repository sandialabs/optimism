from jax import jit
from optimism import EquationSolver
from optimism import VTKWriter
from optimism import FunctionSpace
from optimism import Interpolants
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism.FunctionSpace import DofManager
from optimism.FunctionSpace import EssentialBC
from optimism.material import Neohookean

import jax.numpy as np
from optimism.inverse import AdjointFunctionSpace
from collections import namedtuple

EnergyFunctions = namedtuple('EnergyFunctions',
                            ['energy_function_coords'])

# simulation parameterized on mesh node coordinates
class CoordinateParameterizedSimulation:

    def __init__(self):
        self.writeOutput = True

        self.quad_rule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        self.ebcs = [
            EssentialBC(nodeSet='bottom_sideset', component=1),

            EssentialBC(nodeSet='right_sideset', component=0),
            EssentialBC(nodeSet='left_sideset', component=0),
        ]

        shearModulus = 0.855 # MPa
        bulkModulus = 1000*shearModulus # MPa
        youngModulus = 9.0*bulkModulus*shearModulus / (3.0*bulkModulus + shearModulus)
        poissonRatio = (3.0*bulkModulus - 2.0*shearModulus) / 2.0 / (3.0*bulkModulus + shearModulus)
        props = {
            'elastic modulus': youngModulus,
            'poisson ratio': poissonRatio,
            'version': 'coupled'
        }
        self.mat_model = Neohookean.create_material_model_functions(props)

        self.eq_settings = EquationSolver.get_settings(max_trust_iters=2500, use_preconditioned_inner_product_for_cg=True)

        self.input_mesh = './snap_cell.exo'

        self.maxForce = -3.0
        self.plot_file = 'force_control_response.npz'
        self.stages = 1
        steps_per_stage = 65
        self.steps = self.stages * steps_per_stage

    def create_field(self, Uu):
        def get_ubcs():
            V = np.zeros(self.mesh.coords.shape)
            return self.dof_manager.get_bc_values(V)

        return self.dof_manager.create_field(Uu, get_ubcs())

    def reload_mesh(self):
        origMesh = ReadExodusMesh.read_exodus_mesh(self.input_mesh)
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(origMesh, order=2, createNodeSetsFromSideSets=True)

        func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        self.dof_manager = DofManager(func_space, 2, self.ebcs)

        surfaceXCoords = self.mesh.coords[self.mesh.nodeSets['top_sideset']][:,0]
        self.tractionArea = np.max(surfaceXCoords) - np.min(surfaceXCoords)

        self.stateNotStored = True
        self.state = []

    def run_simulation(self):

        func_space = FunctionSpace.construct_function_space(self.mesh, self.quad_rule)
        mech_funcs = Mechanics.create_mechanics_functions(func_space, mode2D='plane strain', materialModel=self.mat_model)

        # methods defined on the fly
        def energy_function(Uu, p):
            U = self.create_field(Uu)
            internal_variables = p.state_data
            strainEnergy = mech_funcs.compute_strain_energy(U, internal_variables)

            F = p.bc_data
            def force_function(x, n):
                return np.array([0.0, F/self.tractionArea])

            loadPotential = Mechanics.compute_traction_potential_energy(
                func_space, U, self.lineQuadRule, self.mesh.sideSets['top_sideset'], 
                force_function)

            return strainEnergy + loadPotential

        def assemble_sparse(Uu, p):
            U = self.create_field(Uu)
            internal_variables = p.state_data
            element_stiffnesses = mech_funcs.compute_element_stiffnesses(U, internal_variables)
            return SparseMatrixAssembler.\
                assemble_sparse_stiffness_matrix(element_stiffnesses, func_space.mesh.conns, self.dof_manager)
    
        def store_force_displacement(Uu, force_val, force, disp):
            U = self.create_field(Uu)

            index = (self.mesh.nodeSets['top_sideset'], 1)

            force.append( force_val )
            disp.append(np.mean(U.at[index].get()))

            with open(self.plot_file,'wb') as f:
                np.savez(f, force=force, displacement=disp)

        def write_vtk_output(Uu, p, step):
            U = self.create_field(Uu)
            plotName = 'output-'+str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='displ', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

            energyDensities = mech_funcs.compute_output_energy_densities_and_stresses(U, p.state_data)[0]
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(func_space, energyDensities)
            writer.add_cell_field(name='strain_energy_density',
                                  cellData=cellEnergyDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.write()

        # problem set up
        Uu = self.dof_manager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mech_funcs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        precond_strategy = Objective.PrecondStrategy(assemble_sparse)
        self.objective = Objective.Objective(energy_function, Uu, p, precond_strategy)

        # loop over load steps
        force = 0.
        fd_force = []
        fd_disp = []

        store_force_displacement(Uu, force, fd_force, fd_disp)
        self.state.append((Uu, p))

        steps_per_stage = int(self.steps / self.stages)
        force_inc = self.maxForce / steps_per_stage
        for step in range(1, steps_per_stage+1):
            print('--------------------------------------')
            print('LOAD STEP ', step)
            force += force_inc
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)

            store_force_displacement(Uu, force, fd_force, fd_disp)
            self.state.append((Uu, p))

            if self.writeOutput:
              write_vtk_output(Uu, p, step + 1)

        self.stateNotStored = False

    # energy functions for computing optimization quantities of interest
    def setup_energy_functions(self):
        shapeOnRef = Interpolants.compute_shapes(self.mesh.parentElement, self.quad_rule.xigauss)

        def energy_function_all_dofs(U, p, coords):
            adjoint_func_space = AdjointFunctionSpace.construct_function_space_for_adjoint(coords, shapeOnRef, self.mesh, self.quad_rule)
            mech_funcs = Mechanics.create_mechanics_functions(adjoint_func_space, mode2D='plane strain', materialModel=self.mat_model)
            ivs = p.state_data
            return mech_funcs.compute_strain_energy(U, ivs)

        def energy_function_coords(Uu, p, coords):
            U = self.create_field(Uu)
            return energy_function_all_dofs(U, p, coords)

        return EnergyFunctions(energy_function_coords)

    def compute_energy_quantities(self, uSteps, pSteps, coordinates, energy_function_coords):
        index = (self.mesh.nodeSets['top_sideset'], 1)

        totalWork = 0.0
        complementaryWork = 0.0
        totalWorkStored = []
        complementaryWorkStored = []
        strainEnergyStored = []
        dissipatedEnergyStored = []
        for step in range(1, self.steps+1):
            Uu = uSteps[step]
            p = pSteps[step]
            U = self.create_field(Uu)
            force = p.bc_data
            disp = np.mean(U.at[index].get())

            Uu_prev = uSteps[step-1]
            p_prev = pSteps[step-1]
            U_prev = self.create_field(Uu_prev)
            force_prev = p_prev.bc_data
            disp_prev = np.mean(U_prev.at[index].get())

            totalWork += 0.5*(force + force_prev)*(disp - disp_prev)
            complementaryWork += 0.5*(force - force_prev)*(disp + disp_prev)

            totalWorkStored.append(totalWork)
            complementaryWorkStored.append(complementaryWork)
            strainEnergyStored.append(energy_function_coords(Uu, p, coordinates))
            dissipatedEnergyStored.append(totalWork - energy_function_coords(Uu, p, coordinates))

        print("\n Quantities of Interest:")
        print(f"total work: {totalWork}")
        print(f"complementary work: {complementaryWork}")
        print(f"strain energy: {energy_function_coords(Uu, p, coordinates)}")
        print(f"dissipated energy: {totalWork - energy_function_coords(Uu, p, coordinates)}")

        with open("energy_histories.npz",'wb') as f:
            np.savez(f, totalWork=totalWorkStored, complementaryWork=complementaryWorkStored, strainEnergy=strainEnergyStored, dissipatedEnergy=dissipatedEnergyStored)

    def compute_qois(self):
        if self.stateNotStored:
            self.run_simulation()

        parameters = self.mesh.coords
        energyFuncs = self.setup_energy_functions()

        uSteps = np.stack([self.state[i][0] for i in range(0, self.steps+1)], axis=0)
        pSteps = [self.state[i][1] for i in range(0, self.steps+1)]

        self.compute_energy_quantities(uSteps, pSteps, parameters, jit(energyFuncs.energy_function_coords)) 



if __name__ == '__main__':
    sim = CoordinateParameterizedSimulation()
    sim.reload_mesh()
    sim.compute_qois()
