import functools
import jax
import jax.numpy as np
import numpy as onp
import pandas as pd
from scipy.sparse import linalg
from scipy.io import savemat
from sksparse import cholmod
from scipy import linalg as lnal
import matplotlib.pyplot as plt

from optimism import Mesh
from optimism import EquationSolver
from optimism import FunctionSpace
from optimism.FunctionSpace import DofManager, EssentialBC
from optimism import Mechanics
from optimism.material import Neohookean
from optimism import Objective
from optimism import VTKWriter
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import Surface
from optimism import ReadExodusMesh
from exodus3 import exodus

# Constant Properties
E      = 5.0   # Young's modulus
nu     = 0.25  # poissons ratio
rho    = 1.0   # density
nModes = 10    # number of modes to solve for

# input exodus mesh
mesh_input = './beamTest_1.exo'
meshOrig = ReadExodusMesh.read_exodus_mesh(mesh_input)
mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(meshOrig, order=2, createNodeSetsFromSideSets=True)
# nodeTol = 1e-8

# boundary conditions, 0: x-direction, 1: y-direction
ebcs = [EssentialBC(nodeSet='xminus_sideset', component=0),
        EssentialBC(nodeSet='xminus_sideset', component=1)]


# material model
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'density': rho}
# Neohookean with variable properties assigned to each element
material = Neohookean.create_material_model_functions(props)

# stuff for FEM calculations, like shape functions etc. 
quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
surfaceQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=1)
lineQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
fs = FunctionSpace.construct_function_space(mesh, quadRule)
dofManager = DofManager(fs, dim=2, EssentialBCs=ebcs)
# dofManager_fullDofs = DofManager(fs, dim=2, EssentialBCs=[])

# define solid and dynamic function
solidMechanics = Mechanics.create_mechanics_functions(fs, 'plane strain', material)
solidDynamics = Mechanics.create_dynamics_functions(fs, 'plane strain', material, Mechanics.NewmarkParameters)

# pull props from the mesh
#props = ReadExodusMesh.read_exodus_mesh_element_properties(mesh_input, ['light_dose','nu'], blockNum=1)

# energy and assembly methods
def compute_potential_energy(Uu, p):
    U = dofManager.create_field(Uu)
    F = p[0]
    internalVariables = p[1]
    traction_function = lambda X, N: np.array([0.0, F/1]) # The  represents the thickness of the beam. Change if needed
    V = Mechanics.compute_traction_potential_energy(fs, U, surfaceQuadRule, 
                                                    mesh.sideSets['xplus_sideset'], traction_function)
    W = solidMechanics.compute_strain_energy(U, internalVariables)
    return W + V

def energy_function_traction_potential(Uu, p):
    F = p[0]
    U = dofManager.create_field(Uu)
    # loadPotential = 
    return Mechanics.compute_traction_potential_energy(fs, U, lineQuadRule, mesh.sideSets['xplus_sideset'], lambda x, n: np.array([0.0, -F/1]))

def compute_kinetic_energy(Vu):
    V = dofManager.create_field(Vu)
    return solidDynamics.compute_output_kinetic_energy(V)

def assemble_sparse_stiffness_matrix(Uu, p):
    U = dofManager.create_field(Uu)
    internalVariables = p.state_data
    elementStiffnesses = solidMechanics.compute_element_stiffnesses(U, internalVariables)
    return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(
        elementStiffnesses, mesh.conns, dofManager)


precondStrategy = Objective.PrecondStrategy(assemble_sparse_stiffness_matrix)

#########################################################################################################
# Solve SM problem in single load step. 
# Here this is zero force
# so really this is solving the problem about and "undeformed state"
# we need to do this to initialize some stuff under the hood.
#
Uu = np.zeros(dofManager.get_unknown_size()) # set initial unknown displacements to zero
state = solidMechanics.compute_initial_state() # initialize internal state variables
force = 0.0 # initialize applied force
ivs = solidMechanics.compute_initial_state()
p = Objective.Params(force, state) # May have to remove state?


objective = Objective.Objective(compute_potential_energy, Uu, p, precondStrategy)

# # now we can solve the eigen problem in the reference configuration
# n = dofManager.get_unknown_size()

# K = linalg.LinearOperator((n, n), lambda V: onp.asarray(objective.hessian_vec(Uu, V))) # stiffness operator
# KFactor = cholmod.cholesky(assemble_sparse_stiffness_matrix(Uu, p))
# approxKinv = linalg.LinearOperator((n, n), KFactor)
# Kinv = linalg.LinearOperator((n, n), lambda V: linalg.cg(K, V, atol=0, rtol=1e-3, M=approxKinv)[0])

# mass_operator = jax.jit(jax.jacrev(compute_kinetic_energy))
# M = linalg.LinearOperator((n, n), lambda V: onp.asarray(mass_operator(V))) # mass operator

# # # actual eigen solve
# # mu, modes = linalg.eigsh(A=M, k=nModes, M=K, Minv=Kinv, which='LA')
# # lambdas = 1/mu
# # # lambdas.sort()

# # indices = np.argsort(lambdas)
# # lambdas, modes = lambdas[indices], modes[:, indices]



# # define the traction force
# F = p[0]

# # calculate the traction potential energy
# test = energy_function_traction_potential(Uu,p)

# # calculate displacement field
# Utest = dofManager.create_field(Uu)

# # calculate gradient of traction potential energy function with respect to displacement field
# gradFunc =  jax.grad(Mechanics.compute_traction_potential_energy,argnums = 1, allow_int=True)

# # test the gradient function with the inputs
# gradForce = gradFunc(fs, Utest, lineQuadRule, mesh.sideSets['xplus_sideset'], lambda x, n: np.array([0.0, -F/1]))
# # gradTest3 = gradTest2(fs, Utest, lineQuadRule, mesh.nodeSets['xplus_nodeset'], lambda x, n: np.array([0.0, -F/1]))

# # remove the known nodal values
# forceNodes = gradForce[dofManager.isUnknown]


# # calculate FRF
# frf = []
# frequency = range(1000)
# for f in range(np.size(frequency)):
#     w = 2 * np.pi * f
#     A = K - (w**2)*M
#     dispU = linalg.cg(A, forceNodes)[0]
#     accU = dispU*(-w**2)

#     reducedaccU = np.delete(accU, np.nonzero(forceNodes)[0])
#     frfTemp = sum(reducedaccU)/np.shape(reducedaccU)[0]
#     # print(frfTemp)
#     frf.append(np.abs(frfTemp))

# file = open('frfTest.txt','w')
# for i in range(np.shape(frf)[0]):
#     file.write(np.array_str(np.array([frequency[i],frf[i]])))
# file.close()

# plt.loglog(frequency,frf)


# # post-processing
# for n in range(nModes):
#     print('Natural frequency %s = %s' % (n + 1, lambdas[n]))

# writer = VTKWriter.VTKWriter(mesh, baseFileName='output-undeformed')
# for n in range(nModes):
#     writer.add_nodal_field('mode-%s' % (n + 1), dofManager.create_field(modes[:, n]), VTKWriter.VTKFieldType.VECTORS)

# writer.write()



#########################################################################################################
# now pre-load in SM in a single load-step
# force is now 2.0
solverSettings = EquationSolver.get_settings()
force = 2.0
p = Objective.param_index_update(p, 0, force) # put current total force in parameter set
Uu, solverSuccess = EquationSolver.nonlinear_equation_solve(objective, Uu, p, solverSettings, useWarmStart=False)

# writer = VTKWriter.VTKWriter(mesh, baseFileName='output-deformed')
# writer.add_nodal_field('displ', dofManager.create_field(Uu), VTKWriter.VTKFieldType.VECTORS)

# solve eigen problem about pre-loaded state
n = dofManager.get_unknown_size()

K = linalg.LinearOperator((n, n), lambda V: onp.asarray(objective.hessian_vec(Uu, V))) # stiffness operator
KFactor = cholmod.cholesky(assemble_sparse_stiffness_matrix(Uu, p))
approxKinv = linalg.LinearOperator((n, n), KFactor)
Kinv = linalg.LinearOperator((n, n), lambda V: linalg.cg(K, V, atol=0, rtol=1e-3, M=approxKinv)[0])

mass_operator = jax.jit(jax.jacrev(compute_kinetic_energy))
M = linalg.LinearOperator((n, n), lambda V: onp.asarray(mass_operator(V))) # mass operator

# # actual eigen solve
# mu, modes = linalg.eigsh(A=M, k=nModes, M=K, Minv=Kinv, which='LA')
# lambdas = 1/mu
# # lambdas.sort()

# indices = np.argsort(lambdas)
# lambdas, modes = lambdas[indices], modes[:, indices]


# define the traction force
F = p[0]

# calculate the traction potential energy
test = energy_function_traction_potential(Uu,p)

# calculate displacement field
Utest = dofManager.create_field(Uu)

# calculate gradient of traction potential energy function with respect to displacement field
gradFunc =  jax.grad(Mechanics.compute_traction_potential_energy,argnums = 1, allow_int=True)

# test the gradient function with the inputs
gradForce = gradFunc(fs, Utest, lineQuadRule, mesh.sideSets['xplus_sideset'], lambda x, n: np.array([0.0, -F/1]))
# gradTest3 = gradTest2(fs, Utest, lineQuadRule, mesh.nodeSets['xplus_nodeset'], lambda x, n: np.array([0.0, -F/1]))

# remove the known nodal values
forceNodes = gradForce[dofManager.isUnknown]


# calculate FRF
frf = []
# frf_test = []
frequency = []
for f in range(0,1000,5):
    w = 2 * np.pi * f
    frequency.append(w)
    A = K - (w**2)*M
    dispUu, exit = linalg.cg(A, forceNodes)
    # print(exit)
    # accUu = dispUu*(-w**2)

    # accU = dofManager.create_field(accUu)
    dispU = dofManager.create_field(dispUu)
    accU = -dispU*(w**2)
    # print(mesh)
    nsetAcc = accU[mesh.nodeSets['xminus_sideset'],:]
    accX = np.mean(np.abs(nsetAcc[:,0]))
    accY = np.mean(np.abs(nsetAcc[:,1]))
    print(f)
    print(accY)

    # accMag = 


    # # delete the acceleration of nodes where load was applied
    # reducedaccU = np.delete(accU, np.nonzero(forceNodes)[0])
    # #calculate average acceleration of the nodes
    # frfTemp = sum(np.abs(reducedaccU))/np.shape(reducedaccU)[0]
    # # add the average acceleration to the frf array
    frf.append(accY)
    # # frf_test.append(np.abs(accU[0]))

# file = open('frfTest.txt','w')
# for i in range(np.shape(frf)[0]):
#     file.write(np.array_str(np.array([frequency[i],frf[i]])))
# file.close()

plt.loglog(frequency,frf)
# plt.loglog(frequency,frf_test)
plt.title('FRF Beam Test')
plt.legend(["loaded"], loc = "upper left")
plt.savefig('test2.png', dpi=600)
plt.close()

# # post-processing
# for n in range(nModes):
#     print('Natural frequency %s = %s' % (n + 1, lambdas[n]))

# for n in range(nModes):
#     writer.add_nodal_field('mode-%s' % (n + 1), dofManager.create_field(modes[:, n]), VTKWriter.VTKFieldType.VECTORS)

# writer.write()
