import jax.numpy as np
import unittest
from unittest import mock

from jax import jit, grad
from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism import QuadratureRule
from optimism.test import MeshFixture
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism.material import J2Plastic, Neohookean, HyperViscoelastic

from matplotlib import pyplot as plt

plotting=True

class MechanicsFunctionsFixture(MeshFixture.MeshFixture):

    def setUp(self):
        dummyDispGrad = np.eye(2)
        self.mesh = self.create_mesh_and_disp(10, 10, [0.,1.], [0.,1.],
                                                lambda x: dummyDispGrad.dot(x))[0]
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=2, createNodeSetsFromSideSets=True)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        G_eq = 0.855 # MPa

        K_eq = 1000*G_eq # MPa - artificially low bulk modulus so that volumetric strain energy doesn't drown out the dissipation (which is deviatoric)
        poissonRatio = (3.0*K_eq - 2.0*G_eq) / 2.0 / (3.0*K_eq + G_eq)
        print(f"\n Poisson's ratio is: {poissonRatio}")

        # poissonRatio = 0.3
        # K_eq = 2.0 * G_eq * (1.0 + poissonRatio) / 3.0 / (1.0 - 2.0 * poissonRatio)


        G_neq_1 = 4.0*G_eq
        tau_1   = 0.1
        props = {
            'equilibrium bulk modulus'     : K_eq,
            'equilibrium shear modulus'    : G_eq,
            'non equilibrium shear modulus': G_neq_1,
            'relaxation time'              : tau_1,
        } 
        self.mat = HyperViscoelastic.create_material_model_functions(props)

        # shearModulus = G_eq # MPa
        # bulkModulus = K_eq # MPa
        # youngModulus = 9.0*bulkModulus*shearModulus / (3.0*bulkModulus + shearModulus)
        # poissonRatio = (3.0*bulkModulus - 2.0*shearModulus) / 2.0 / (3.0*bulkModulus + shearModulus)
        # props = {
        #     'elastic modulus': youngModulus,
        #     'poisson ratio': poissonRatio,
        #     'version': 'coupled'
        # }
        # self.matNeoHooke = Neohookean.create_material_model_functions(props)

        self.EBCs = [FunctionSpace.EssentialBC(nodeSet='top', component=0),
                FunctionSpace.EssentialBC(nodeSet='top', component=1),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        # totalTime = 1.0e1*tau_1
        # self.stages = 2
        # self.steps_per_stage = 50
        # self.steps = self.stages*self.steps_per_stage
        # self.dt = totalTime / self.steps_per_stage
        # self.maxDisp = 1.0

        self.dt = tau_1/10
        self.steps_per_stage = 20
        self.stages = 2
        self.steps = self.stages*self.steps_per_stage
        self.maxDisp = 1.0

    def test_integrate_material_qoi_using_dissipated_energy(self):
        mechFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                         "plane strain",
                                                         self.mat)
        dofManager = FunctionSpace.DofManager(self.fs, 2, self.EBCs)

        def create_field(Uu, disp):
            def get_ubcs(disp):
                V = np.zeros(self.mesh.coords.shape)
                index = (self.mesh.nodeSets['top'], 1)
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
        integrate_free_energy = jit(compute_energy)

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

        # Forward solve
        Uu = dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        U = create_field(Uu, p.bc_data)
        self.objective = Objective.Objective(compute_energy, Uu, p)

        index = (self.mesh.nodeSets['top'], 1)

        disp_inc = self.maxDisp / self.steps
        disp = 0.0
        externalWorkStore = []
        dissipationStore = []
        freeEnergyStore = []
        for step in range(1, self.steps+1):
            force_prev = np.array(nodal_forces(U, p).at[index].get())
            applied_disp_prev = U.at[index].get()

            if step <= self.steps_per_stage:
                disp += disp_inc
            else:
                disp -= disp_inc

            p = Objective.param_index_update(p, 0, disp)
            Uu = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)

            write_output(Uu, p, step)
            
            force = np.array(nodal_forces(U, p).at[index].get())
            applied_disp = U.at[index].get()
            externalWorkStore.append( 0.5*np.tensordot((force + force_prev),(applied_disp - applied_disp_prev), axes=1) )
            freeEnergyStore.append(integrate_free_energy(Uu, p))
            # SHOULD USE IVS_PREV?
            dissipationStore.append( self.dt * integrate_dissipation(U, ivs, self.dt) )

        if plotting:
            plt.figure()
            plt.plot(np.cumsum(np.array(externalWorkStore)), marker='o', fillstyle='none')
            plt.plot(np.array(freeEnergyStore), marker='x', fillstyle='none')
            plt.plot(np.cumsum(np.array(dissipationStore)), marker='v', fillstyle='none')
            plt.plot(np.cumsum(np.array(dissipationStore)) + np.array(freeEnergyStore), marker='*', fillstyle='none')
            # plt.xlabel('Time')
            plt.xlabel('Step')
            plt.ylabel('Energy')
            plt.legend(["External", "Free", "Dissipated", "Free + Dissipated"], loc=0, frameon=True)
            plt.savefig('energy_histories.png')

    @unittest.skip("Debugging other tests")
    def test_sanity_check_hyperelastic(self):
        mechFuncs = Mechanics.create_mechanics_functions(self.fs,
                                                         "plane strain",
                                                         self.matNeoHooke)
        dofManager = FunctionSpace.DofManager(self.fs, 2, self.EBCs)

        def create_field(Uu, disp):
            def get_ubcs(disp):
                V = np.zeros(self.mesh.coords.shape)
                index = (self.mesh.nodeSets['top'], 1)
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
        integrate_free_energy = jit(compute_energy)

        # Forward solve
        Uu = dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        U = create_field(Uu, p.bc_data)
        self.objective = Objective.Objective(compute_energy, Uu, p)

        index = (self.mesh.nodeSets['top'], 1)

        disp_inc = self.maxDisp / self.steps
        disp = 0.0
        externalWorkStore = []
        freeEnergyStore = []
        for step in range(1, self.steps+1):
            force_prev = np.array(nodal_forces(U, p).at[index].get())
            applied_disp_prev = U.at[index].get()

            if step <= self.steps_per_stage:
                disp += disp_inc
            else:
                disp -= disp_inc

            p = Objective.param_index_update(p, 0, disp)
            Uu = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, EqSolver.get_settings(), useWarmStart=False)
            U = create_field(Uu, p.bc_data)
            ivs = mechFuncs.compute_updated_internal_variables(U, p.state_data, self.dt)
            p = Objective.param_index_update(p, 1, ivs)
            
            force = np.array(nodal_forces(U, p).at[index].get())
            applied_disp = U.at[index].get()
            externalWorkStore.append( 0.5*np.tensordot((force + force_prev),(applied_disp - applied_disp_prev), axes=1) )
            freeEnergyStore.append(integrate_free_energy(Uu, p))

        if plotting:
            plt.figure()
            plt.plot(np.cumsum(np.array(externalWorkStore)), marker='o', fillstyle='none')
            plt.plot(np.array(freeEnergyStore), marker='x', fillstyle='none')
            # plt.xlabel('Time')
            plt.xlabel('Step')
            plt.ylabel('Energy')
            plt.legend(["External", "Free"], loc=0, frameon=True)
            plt.savefig('energy_histories.png')



class MultiBlockMechanicsFunctionsFixture(MeshFixture.MeshFixture):

    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.0, 1.0]
        yRange = [0.0, 1.0]

        self.targetDispGrad = np.array([[0.1, -0.2], [0.4, -0.1]])

        mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                 lambda x: self.targetDispGrad.dot(x))
        blocks = {'block0': np.array([0, 1, 2, 3]),
                  'block1': np.array([4, 5, 6, 7])}
        self.mesh = Mesh.mesh_with_blocks(mesh, blocks)
        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        props = {'elastic modulus': 1,
                 'poisson ratio': 0.25,
                 'yield strength': 0.1,
                 'hardening model': 'linear',
                 'hardening modulus': 0.1}

        materialModel0 = Neohookean.create_material_model_functions(props)
        materialModel1 = J2Plastic.create_material_model_functions(props)
        self.blockModels = {'block0': materialModel0, 'block1': materialModel1}

    @unittest.skip("Debugging other tests")
    def test_internal_variables_initialization(self):
        nQuadPoints = QuadratureRule.len(self.quadRule)
        internals = Mechanics._compute_initial_state_multi_block(self.fs, self.blockModels)
        self.assertEqual(internals.shape, (Mesh.num_elements(self.mesh), nQuadPoints, 10))
        self.assertArrayEqual(internals[0, 0], np.zeros(J2Plastic.NUM_STATE_VARS))
        self.assertArrayEqual(internals[4, 0], np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))

    @unittest.skip("Debugging other tests")
    def test_internal_variables_update(self):
        internals = Mechanics._compute_initial_state_multi_block(self.fs, self.blockModels)
        dt = 1.0
        internalsNew = Mechanics._compute_updated_internal_variables_multi_block(self.fs, self.U, internals, dt, self.blockModels, Mechanics.plane_strain_gradient_transformation)
        self.assertEqual(internals.shape, internalsNew.shape)
        self.assertGreater(internalsNew[4,0,J2Plastic.EQPS], 0.05)

if __name__ == "__main__":
    unittest.main()
