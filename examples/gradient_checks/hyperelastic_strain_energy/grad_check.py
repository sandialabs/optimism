import sys
import os
import exodus
from optimism import ReadExodusMesh
import finite_difference_utils as fd
import NodalCoordinateOptimization

# specify sensitivity check parameters
model = "window"

nodeSets = {"inner_nodeset"}

iniStepSize = float(sys.argv[1])
numSteps    = int(sys.argv[2])

# create optimism class instance
optimism = NodalCoordinateOptimization.NodalCoordinateOptimization()

# compute objective value
optimism.reload_mesh()
fval = optimism.get_objective()

# compute Df_DX
DFDX = optimism.get_gradient()

# save original mesh
meshName = model + ".exo"
originalMeshName = fd.save_original_mesh(meshName)

# function: get node set nodes
def getNodeSetNodes(nodesets):

    meshWithNsets = ReadExodusMesh.read_exodus_mesh(meshName)

    tNodes = []
    tFound = False
    for nodeset in nodesets:

        # for tNodeSet in mesh.NodeSets:
        for name in meshWithNsets.nodeSets:
            if name == nodeset:
                vals = meshWithNsets.nodeSets[name]
                for node in vals:
                    tNodes.append(node)
                tFound = True

        if tFound == False:
            raise Exception("Node set '" + nodeset + "' not found")

        if len(tNodes) == 0:
            raise Exception("Found empty node set")

    return tNodes

# function: inner product
def inner_product(vec1,vec2):
    if len(vec1) != len(vec2):
        raise Exception("Vectors for inner product have different lengths!")
    
    result = 0
    for ord in range(0,len(vec1)):
        result += vec1[ord] * vec2[ord]

    return result

# function: get nodal errors with finite difference comparison
def get_nodal_errors(outFile):

    vector = fd.build_direction_vector(mesh.numNodes, nodeSetNodes)
    fd_val = fd.forward_difference(optimism, fval, originalMeshName, meshName, stepSize, vector)

    inner_prod = inner_product(vector, DFDX)

    error = abs(inner_prod - fd_val)

    # write to file
    outFile.write("%3.6E \t" % inner_prod)
    outFile.write("%3.6E \t" % fd_val)
    outFile.write("%3.6E \n" % error)

# loop through different FD step sizes
stepSize = iniStepSize

outFile = open("gradient_check.out","w")
outFile.write("Step Size \t grad'*dir \t FD Approx \t abs(Error) \n")
outFile.write("--------- \t --------- \t --------- \t ---------- \n")

# get node set nodes
mesh = exodus.ExodusDB()
mesh.read(originalMeshName)
nodeSetNodes = getNodeSetNodes(nodeSets)

for step in range(0,numSteps):

    # compute FD approximation for gradients of all nodes
    outFile.write("%3.4E \t" % stepSize)

    get_nodal_errors(outFile)

    # copy back original mesh
    copyStr = "cp " + originalMeshName + " " + meshName
    os.system(copyStr)

    stepSize *= 1.0e-1

# finalize
outFile.close()
