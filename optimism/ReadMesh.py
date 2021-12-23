import json

from optimism.JaxConfig import *
from optimism import Mesh

def read_json_mesh(meshFileName):

    with open(meshFileName, 'r', encoding='utf-8') as jsonFile:
        meshData = json.load(jsonFile)
        
    coordinates = np.array(meshData['coordinates'])
    connectivity = np.array(meshData['connectivity'], dtype=int)
    nodeSets = {}
    for key in meshData['nodeSets']:
        nodeSets[key] = np.array(meshData['nodeSets'][key])

    sideSets = {}
    exodusSideSets = meshData['sideSets']
    for key in exodusSideSets:
        elements = np.array(exodusSideSets[key][0], dtype=int)
        sides = np.array(exodusSideSets[key][1], dtype=int)
        sideSets[key] = np.column_stack((elements, sides))

    blocks=None
        
    return Mesh.construct_mesh_from_basic_data(coordinates, connectivity,
                                               blocks, nodeSets, sideSets)

