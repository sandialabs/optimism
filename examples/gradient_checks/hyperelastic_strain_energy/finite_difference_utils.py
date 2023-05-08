import exodus
import numpy as np

def forward_difference(optimism, fval, originalMeshName, meshName, step, vector):

    # forward step
    perturb_mesh(originalMeshName, meshName, step, vector)
    optimism.reload_mesh()
    f_plus = optimism.get_objective()

    # compute fd value
    DfDX = (f_plus - fval) / step

    return DfDX

def perturb_mesh(originalMeshName, meshName, step, vector):

    # read in mesh
    mesh = exodus.ExodusDB()
    mesh.read(originalMeshName)

    # perturb coords
    coordsX = mesh.elementBlocks[0].coordinates[0]
    coordsY = mesh.elementBlocks[0].coordinates[1]

    vecX = vector[0::2]
    vecY = vector[1::2]

    for ord in range(0, len(coordsX)):
        coordsX[ord] += step * vecX[ord]
        coordsY[ord] += step * vecY[ord]

    mesh.write(meshName)

def save_original_mesh(meshName):
    mesh = exodus.ExodusDB()
    mesh.read(meshName)

    savedName = meshName + ".orig"
    mesh.write(savedName)
    return savedName

def build_direction_vector(numNodes, nodeSetNodes):

    numDims = 2
    numNodeSetNodes = len(nodeSetNodes)

    # random direction vector for node set nodes
    np.random.seed(123)
    directionVector = np.random.uniform(0.0, 1.0, numDims*numNodeSetNodes)
    normVector = directionVector / np.linalg.norm(directionVector)

    # fill in values for all nodes (0 if not in node set)
    outVector = [0] * numDims * numNodes
    for nodeOrd in range(numNodeSetNodes):
        nodeID = nodeSetNodes[nodeOrd]
        outVector[numDims*nodeID + 0] = normVector[numDims*nodeOrd + 0]
        outVector[numDims*nodeID + 1] = normVector[numDims*nodeOrd + 1]

    return outVector