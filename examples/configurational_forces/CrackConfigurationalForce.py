import jax
import jax.numpy as np

from optimism import EquationSolver
from optimism import Interpolants
from optimism import FunctionSpace
from optimism.material import LinearElastic
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism import QuadratureRule
from optimism import VTKWriter

# set up the mesh
mesh = ReadExodusMesh.read_exodus_mesh('crack_domain.g')

kFieldXOffset = mesh.coords[mesh.nodeSets['crack_tip'], 0]

quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(mesh.parentElement.degree - 1))
shapeData = Interpolants.compute_shapes(mesh.parentElement, quadRule.xigauss)
fs = FunctionSpace.construct_function_space_from_parent_element(mesh, shapeData, quadRule)

ebcs = [FunctionSpace.EssentialBC(nodeSet='external', component=0),
        FunctionSpace.EssentialBC(nodeSet='external', component=1),
        FunctionSpace.EssentialBC(nodeSet='ysymm', component=1)]

essentialBCManager = FunctionSpace.DofManager(fs, dim=2, EssentialBCs=ebcs)

E = 1.0
nu = 0.25
G = 0.5*E/(1 + nu)
props = {"elastic modulus": E,
         "poisson ratio": nu}
material = LinearElastic.create_material_model_functions(props)

solidMechanics = Mechanics.create_mechanics_functions(fs, "plane strain", material)

def apply_mode_I_field_at_point(X, K_I):
    R = np.linalg.norm(X)
    theta = np.arctan2(X[1], X[0])    
    u0 = K_I/G*np.sqrt(R/2.0/np.pi)*(1.0 - 2.0*nu + np.sin(0.5*theta)**2)*np.cos(0.5*theta)
    u1 = K_I/G*np.sqrt(R/2.0/np.pi)*(2.0 - 2.0*nu - np.cos(0.5*theta)**2)*np.sin(0.5*theta)
    return np.array([u0, u1])

def apply_mode_I_field(coords, K_I):
    return jax.vmap(apply_mode_I_field_at_point, (0, None))(coords, K_I)

def get_ubcs(p):
    appliedK = p[0]
    X = mesh.coords[mesh.nodeSets['external'], :]
    X = X.at[:, 0].add(-kFieldXOffset)
    U = apply_mode_I_field(X, appliedK)
    V = np.zeros_like(mesh.coords).at[mesh.nodeSets['external'], :].set(U)
    return essentialBCManager.get_bc_values(V)

def create_field(Uu, p):
    return essentialBCManager.create_field(Uu, get_ubcs(p))

# write the energy to minimize
def potential_energy(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p[1]
    return solidMechanics.compute_strain_energy(U, internalVariables)

def assemble_sparse_preconditioner_matrix(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p[1]
    elementStiffnesses = solidMechanics.compute_element_stiffnesses(U, internalVariables)
    return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(
        elementStiffnesses, mesh.conns, essentialBCManager)

def write_output(U, p, S, step):
    vtkFilename = "crack_mode_1-" + str(step).zfill(3)
    writer = VTKWriter.VTKWriter(mesh, baseFileName=vtkFilename)
    
    writer.add_nodal_field("bcs", nodalData=np.array(essentialBCManager.isBc, dtype=int),
                           fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)
    
    writer.add_nodal_field("displacement", nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

    internalVariables = p[1]
    energyDensities, stresses = solidMechanics.compute_output_energy_densities_and_stresses(
        U, internalVariables)
    cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(
        fs, energyDensities)
    cellStresses = FunctionSpace.project_quadrature_field_to_element_field(
        fs, stresses)
    
    writer.add_cell_field("strain_energy_density", cellEnergyDensities, VTKWriter.VTKFieldType.SCALARS)
    writer.add_cell_field("stress", cellStresses, VTKWriter.VTKFieldType.TENSORS)
    
    writer.add_nodal_field("config_forces", nodalData=S, fieldType=VTKWriter.VTKFieldType.VECTORS)
    
    # double the configurational force to account for half-symmetric model
    J = -S[mesh.nodeSets['crack_tip'], 0].item() * 2
    print('Configurational force on crack tip:', J)
    K = p[0]
    print('Applied energy release rate:', K**2 / (E/(1 - nu**2)))
    
    writer.write()

def qoi(Wu, Uu, p, mesh):
    W = essentialBCManager.create_field(Wu)
    coords = mesh.coords + W
    qoiMesh = Mesh.mesh_with_coords(mesh, coords)
    qoiFS = FunctionSpace.construct_function_space_from_parent_element(qoiMesh, shapeData, quadRule)
    qoiSolidMechanics = Mechanics.create_mechanics_functions(qoiFS, "plane strain", material)
    internalVariables = p[1]
    U = create_field(Uu, p)
    return qoiSolidMechanics.compute_strain_energy(U, internalVariables)
    
sensit = jax.jit(jax.value_and_grad(qoi, 0))

solverSettings = EquationSolver.get_settings(max_cumulative_cg_iters=100,
                                             max_trust_iters=100,
                                             use_preconditioned_inner_product_for_cg=True)

precondStrategy = Objective.PrecondStrategy(assemble_sparse_preconditioner_matrix)

def run():
    U = np.zeros_like(mesh.coords)
    Uu = essentialBCManager.get_unknown_values(U)
    appliedK = 0.0
    state = solidMechanics.compute_initial_state()

    p = Objective.Params(appliedK, state)
    objective = Objective.Objective(potential_energy, Uu, p, precondStrategy)

    for i in range(1):
        appliedK += 1.0
        p = Objective.param_index_update(p, 0, appliedK)

        Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, solverSettings)

        
        Wu = np.zeros_like(Uu)
        _, configForcesOnUnknowns = sensit(Wu, Uu, p, mesh)
        U = create_field(Uu, p)
        configForces = essentialBCManager.create_field(configForcesOnUnknowns)
        write_output(U, p, configForces, i)

if __name__ == "__main__":
    run()