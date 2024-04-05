import jax.numpy as np
from jax import jit, grad

from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import QuadratureRule
from optimism.test.MeshFixture import MeshFixture
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism.material import HyperViscoelastic

# This example recreates the shear test considered in
#   Reese and Govindjee (1998). A theory of finite viscoelasticity and numerical aspects. 
#       https://doi.org/10.1016/S0020-7683(97)00217-5
# and in variational minimization form in 
#   Fancello, Ponthot and Stainier (2006). A variational formulation of constitutive models and updates in non-linear finite viscoelasticity.
#        https://doi.org/10.1002/nme.1525

class ShearTest(MeshFixture):

    def setUp(self):
        dummyDispGrad = np.eye(2)
        self.mesh = self.create_mesh_and_disp(3, 3, [0.,1.], [0.,1.],
                                                lambda x: dummyDispGrad.dot(x))[0]
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=2, createNodeSetsFromSideSets=True)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        G_eq = 30.25 
        K_eq = 1000*G_eq 
        G_neq_1 = 77.77 
        tau_1   = 17.5 
        props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        self.mat = HyperViscoelastic.create_material_model_functions(props)

        self.EBCs = [FunctionSpace.EssentialBC(nodeSet='top', component=1),
                FunctionSpace.EssentialBC(nodeSet='top', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

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
        dofManager = FunctionSpace.DofManager(self.fs, 2, self.EBCs)

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
        integrate_dissipation = jit(mechFuncs.integrated_material_qoi)
        integrate_free_energy = jit(mechFuncs.compute_strain_energy)

        def write_output(Uu, p, step):
            from optimism import VTKWriter
            U = create_field(Uu, p.bc_data)
            plotName = 'mechanics-'+str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='displ', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

            energyDensities, stresses = mechFuncs.\
                compute_output_energy_densities_and_stresses(U, p.state_data, self.dt)
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, energyDensities)
            cellStresses = FunctionSpace.project_quadrature_field_to_element_field(self.fs, stresses)
            dissipationDensities = mechFuncs.compute_output_material_qoi(U, p.state_data, self.dt)
            cellDissipationDensities = FunctionSpace.project_quadrature_field_to_element_field(self.fs, dissipationDensities)
            writer.add_cell_field(name='strain_energy_density',
                                  cellData=cellEnergyDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.add_cell_field(name='piola_stress',
                                  cellData=cellStresses,
                                  fieldType=VTKWriter.VTKFieldType.TENSORS)
            writer.add_cell_field(name='dissipation_density',
                                  cellData=cellDissipationDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.write()

        Uu = dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        U = create_field(Uu, p.bc_data)
        self.objective = Objective.Objective(compute_energy, Uu, p)

        index = (self.mesh.nodeSets['top'], 0)

        time = 0.0
        times = []
        externalWorkStore = []
        dissipationStore = []
        incrementalPotentialStore = []
        forceHistory = []
        dispHistory = []
        for step in range(1, self.steps+1):
            force_prev = np.array(nodal_forces(U, p).at[index].get())
            applied_disp_prev = U.at[index].get()

            disp = self.maxDisp * np.sin(2.0 * np.pi * self.freq * time)

            p = Objective.param_index_update(p, 0, disp)
            Uu = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
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

            dissipationStore.append( integrate_dissipation(U, ivs, self.dt) )

            times.append(time)
            time += self.dt

        # storing for plots
        with open("energy_histories.npz",'wb') as f:
            np.savez(f, externalWork=np.cumsum(np.array(externalWorkStore)), dissipation=np.cumsum(np.array(dissipationStore)), algorithmicPotential=np.array(incrementalPotentialStore), time=np.array(times))

        with open("force_disp_histories.npz",'wb') as f:
            np.savez(f, forces=np.array(forceHistory), disps=np.array(dispHistory))



app = ShearTest()
app.setUp()
app.run()