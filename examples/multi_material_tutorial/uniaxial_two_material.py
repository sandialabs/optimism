import numpy as onp # as is "old numpy"

from optimism.JaxConfig import *
from optimism import EquationSolver
from optimism import FunctionSpace
from optimism.material import J2Plastic
from optimism.material import Neohookean
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

        
# set up the mesh
w = 0.1
L = 1.0
nodesInX = 5 # must be odd
nodesInY = 15
xRange = [0.0, w]
yRange = [0.0, L]
mesh = Mesh.construct_structured_mesh(nodesInX, nodesInY, xRange, yRange)

def triangle_centroid(vertices):
    return np.average(vertices, axis=0)

blockIds = -1*onp.ones(Mesh.num_elements(mesh), dtype=onp.int64)
for e, t in enumerate(mesh.conns):
    vertices = mesh.coords[t,:]
    if triangle_centroid(vertices)[0] < w/2:
        blockIds[e] = 0
    else:
        blockIds[e] = 1

blocks = {'soft': np.flatnonzero(np.array(blockIds) == 0),
          'hard': np.flatnonzero(np.array(blockIds) == 1)}

mesh = Mesh.mesh_with_blocks(mesh, blocks)

nodeTol = 1e-8
nodeSets = {'left': np.flatnonzero(mesh.coords[:,0] < xRange[0] + nodeTol),
            'right': np.flatnonzero(mesh.coords[:,0] > xRange[1] - nodeTol),
            'bottom': np.flatnonzero(mesh.coords[:,1] < yRange[0] + nodeTol),
            'top': np.flatnonzero(mesh.coords[:,1] > yRange[1] - nodeTol)}

mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)

# create the function space
order = 2*mesh.masterElement.degree
quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(order - 1))
fs = FunctionSpace.construct_function_space(mesh, quadRule)

# set the essential boundary conditions and create the a DofManager to
# handle the indexing between unknowns and degrees of freedom.
ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
        FunctionSpace.EssentialBC(nodeSet='bottom', component=1),
        FunctionSpace.EssentialBC(nodeSet='top', component=1)]

dofManager = FunctionSpace.DofManager(fs, dim=2, EssentialBCs=ebcs)

# write blocks and bcs to paraview output to check things are correct
writer = VTKWriter.VTKWriter(mesh, baseFileName='check_problem_setup')

writer.add_cell_field(name='block_id', cellData=blockIds,
                      fieldType=VTKWriter.VTKFieldType.SCALARS,
                      dataType=VTKWriter.VTKDataType.INT)

bcs = np.array(dofManager.isBc, dtype=np.int64)
writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS,
                       dataType=VTKWriter.VTKDataType.INT)
    
writer.write()


# create the material models
ESoft = 5.0
nuSoft = 0.25
props = {'elastic modulus': ESoft,
         'poisson ratio': nuSoft}
softMaterialModel = Neohookean.create_material_model_functions(props)

E = 10.0
nu = 0.25
Y0 = 0.01*E
H = E/100
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'yield strength': Y0,
         'hardening model': 'linear',
         'hardening modulus': H}
hardMaterialModel = J2Plastic.create_material_model_functions(props)

materialModels = {'soft': softMaterialModel, 'hard': hardMaterialModel}
    
# mechanics functions
mechanicsFunctions = Mechanics.create_multi_block_mechanics_functions(fs, mode2D="plane strain", materialModels=materialModels)

# helper function to fill in nodal values of essential boundary conditions
def get_ubcs(p):
    appliedDisp = p[0]
    EbcIndex = (mesh.nodeSets['top'], 1)
    V = np.zeros_like(mesh.coords).at[EbcIndex].set(appliedDisp)
    return dofManager.get_bc_values(V)

# helper function to go from unknowns to full DoF array
def create_field(Uu, p):
    return dofManager.create_field(Uu, get_ubcs(p))
    
# write the energy to minimize
def energy_function(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p.state_data
    return mechanicsFunctions.compute_strain_energy(U, internalVariables)
    
# Tell objective how to assemble preconditioner matrix
def assemble_sparse_preconditioner_matrix(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p.state_data
    elementStiffnesses = mechanicsFunctions.compute_element_stiffnesses(U, internalVariables)
    return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(
        elementStiffnesses, mesh.conns, dofManager)

# solver settings
solverSettings = EquationSolver.get_settings(max_cumulative_cg_iters=100,
                                             max_trust_iters=1000,
                                             use_preconditioned_inner_product_for_cg=True)

precondStrategy = Objective.PrecondStrategy(assemble_sparse_preconditioner_matrix)

# initialize unknown displacements to zero
Uu = np.zeros(dofManager.get_unknown_size())

# set initial values of parameters
appliedDisp = 0.0
state = mechanicsFunctions.compute_initial_state()
p = Objective.Params(appliedDisp, state)
    
# Construct an objective object for the equation solver to work on
objective = Objective.ScaledObjective(energy_function, Uu, p, precondStrategy=precondStrategy)
    
# increment the applied displacement
appliedDisp = L*0.003
p = Objective.param_index_update(p, 0, appliedDisp)

# Find unknown displacements by minimizing the objective
Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, solverSettings)

# update the state variables in the new equilibrium configuration
U = create_field(Uu, p)
state = mechanicsFunctions.compute_updated_internal_variables(U, p.state_data)
p = Objective.param_index_update(p, 1, state)

# write solution data to VTK file for post-processing
writer = VTKWriter.VTKWriter(mesh, baseFileName='uniaxial_two_material')

U = create_field(Uu, p)
writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

_, stresses = mechanicsFunctions.compute_output_energy_densities_and_stresses(
    U, state)
cellStresses = FunctionSpace.project_quadrature_field_to_element_field(fs, stresses)
writer.add_cell_field(name='stress', cellData=cellStresses,
                      fieldType=VTKWriter.VTKFieldType.TENSORS)

writer.write()
