# --------------------------------------------------------------------------
# TEST CASE - Compression Test for implementation of MPCs
# --------------------------------------------------------------------------
# Note - Using Cubit Coreform for meshing, solved using NON-LINEAR SOLVER
# --------------------------------------------------------------------------
import sys
sys.path.insert(0, "/home/sarvesh/Documents/Github/optimism")

from jax import jit
from optimism import VTKWriter
from optimism import EquationSolver
from optimism import FunctionSpace
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism.FunctionSpace import DofManagerMPC
from optimism.FunctionSpace import EssentialBC
from optimism.material import Neohookean

import jax.numpy as np
import numpy as onp

class CompressionTest:

    def __init__(self):
        self.writeOutput = True

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        self.EBCs = [
            EssentialBC(nodeSet='bottom', component=0),
            EssentialBC(nodeSet='bottom', component=1)
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

        self.matModel = Neohookean.create_material_model_functions(props)

        self.input_mesh = 'square_mesh_compression.exo'

        self.maxForce = -0.2
        self.steps = 50

    def create_field(self, Uu):
        def get_ubcs():
            V = np.zeros(self.mesh.coords.shape)
            return self.dofManager.get_bc_values(V)
        
        return self.dofManager.create_field(Uu, get_ubcs())
    
    def reload_mesh(self):
        origMesh = ReadExodusMesh.read_exodus_mesh(self.input_mesh)
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(origMesh, order=2, createNodeSetsFromSideSets=True)
        coords = self.mesh.coords
        tol = 1e-8
        # self.mesh.nodeSets = {}

        # Creating nodesets for master and slave
        self.mesh.nodeSets['master'] = np.flatnonzero(coords[:,0] < -0.5 + tol)
        self.mesh.nodeSets['slave'] = np.flatnonzero(coords[:,0] > 0.5 - tol)

        funcSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        self.dofManager = DofManagerMPC(funcSpace, 2, self.EBCs, self.mesh)

        surfaceXCoords = self.mesh.coords[self.mesh.nodeSets['top']][:,0]
        self.tractionArea = np.max(surfaceXCoords) - np.min(surfaceXCoords)

    def run(self):
        funcSpace = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        mechFuncs = Mechanics.create_mechanics_functions(funcSpace, mode2D='plane strain', materialModel=self.matModel)

        def energy_function(Uu, p):
            U = self.create_field(Uu)
            internal_variables = p.state_data
            strainEnergy = mechFuncs.compute_strain_energy(U, internal_variables)

            F = p.bc_data
            def force_function(x, n):
                return np.array([0.0, F/self.tractionArea])
            
            loadPotential = Mechanics.compute_traction_potential_energy(
                funcSpace, U, self.lineQuadRule, self.mesh.sideSets['top'],
                force_function)
            
            return strainEnergy + loadPotential
        
        def write_output(Uu, p, step):
            U = self.create_field(Uu)
            plotName = 'results/' + 'output-' + str(step).zfill(3)
            writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

            writer.add_nodal_field(name='disp', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)
            energyDensities, stresses = mechFuncs.compute_output_energy_densities_and_stresses(U, p.state_data)
            cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(funcSpace, energyDensities)
            cellStresses = FunctionSpace.project_quadrature_field_to_element_field(funcSpace, stresses)
            writer.add_cell_field(name='strain_energy_density',
                                  cellData=cellEnergyDensities,
                                  fieldType=VTKWriter.VTKFieldType.SCALARS)
            writer.add_cell_field(name='cauchy_stress',
                                  cellData=cellStresses,
                                  fieldType=VTKWriter.VTKFieldType.TENSORS)
            writer.write()

        # Problem Set Up 
        Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        ivs = mechFuncs.compute_initial_state()
        p = Objective.Params(bc_data=0., state_data=ivs)
        self.objective = Objective.ObjectiveMPC(energy_function, Uu, p, self.dofManager)

        # Load over the steps
        force = 0.

        force_inc = self.maxForce / self.steps
        
        # K = self.objective.hessian(Uu)
        # print(f"Hessian first row: {K[0,:5]}") 
        # print(f"Hessian Shape: {K.shape}")

        for step in range(1, self.steps):
            print('------------------------------------')
            print('LOAD STEP', step)
            force += force_inc
            p = Objective.param_index_update(p, 0, force)
            Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(self.objective, Uu, p, EquationSolver.get_settings())

            if self.writeOutput:
                write_output(Uu, p, step)




if __name__ == '__main__':
    app = CompressionTest()  
    app.reload_mesh()  
    app.run() 





