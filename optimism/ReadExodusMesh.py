import netCDF4

from optimism.JaxConfig import *
from optimism import Mesh
from optimism import Interpolants

exodusToNativeTri6NodeOrder = np.array([0, 3, 1, 5, 4, 2])


def read_exodus_mesh(fileName):
    with netCDF4.Dataset(fileName) as exData:
        coords = _read_coordinates(exData)
        conns, blocks = _read_blocks(exData)
        nodeSets = _read_node_sets(exData)
        sideSets = _read_side_sets(exData)

        elementType = _read_element_type(exData).lower()
        if elementType == "tri3" or elementType == "tri":
            basis, basis1d = Interpolants.make_parent_elements(degree = 1)
            simplexNodesOrdinals = np.arange(coords.shape[0])
        elif elementType == "tri6":
            basis, basis1d = Interpolants.make_parent_elements(degree = 2)
            simplexNodesOrdinals = _get_vertex_nodes_from_exodus_tri6_mesh(conns)
            conns = conns[:, exodusToNativeTri6NodeOrder]
        else:
            raise

        return Mesh.Mesh(coords, conns, simplexNodesOrdinals, basis, basis1d,
                         blocks, nodeSets, sideSets)
        
        
def _read_coordinates(exodusDataset):
    nNodes = len(exodusDataset.dimensions['num_nodes'])
    nDims = len(exodusDataset.dimensions['num_dim'])
    assert nDims == 2

    coordsX = exodusDataset.variables['coordx'][:]
    coordsY = exodusDataset.variables['coordy'][:]
    
    return np.column_stack([coordsX, coordsY])


def _read_block_conns(exodusDataset, blockOrdinal):
    key = 'connect' + str(blockOrdinal + 1)
    record = exodusDataset.variables[key]
    record.set_auto_mask(False)

    nElemsInBlock = len(exodusDataset.dimensions['num_el_in_blk' + str(blockOrdinal + 1)])
    return np.array(record[:] - 1)


def _read_blocks(exodusDataset):
    nodesPerElem = len(exodusDataset.dimensions['num_nod_per_el1'])

    blockNames = _read_names_list(exodusDataset, 'eb_names')
    # give unnamed blocks an auto-generated name
    for i, name in enumerate(blockNames):
        if not name:
            blockNames[i] = "block_" + str(i+1)
            
    nBlocks = len(exodusDataset.dimensions['num_el_blk'])
    blockConns = []
    blocks = {}
    firstElemInBlock = 0
    for i in range(nBlocks):
        nodesPerElemInBlock = len(exodusDataset.dimensions['num_nod_per_el' + str(i+1)])
        assert nodesPerElemInBlock == nodesPerElem

        blockConns.append(_read_block_conns(exodusDataset, i))
        
        nElemsInBlock = len(exodusDataset.dimensions['num_el_in_blk' + str(i + 1)])
        elemRange = np.arange(firstElemInBlock, firstElemInBlock + nElemsInBlock)
        blocks[blockNames[i]] = elemRange
        firstElemInBlock += nElemsInBlock

    conns = np.vstack(blockConns)
    return conns, blocks


def _read_node_sets(exodusDataset):
    if "num_node_sets" in exodusDataset.dimensions:
        nodeSetNames = _read_names_list(exodusDataset, "ns_names")
        for i, name in enumerate(nodeSetNames):
            if not name:
                nodeSetNames[i] = "nodeset_" + str(i+1)
            
        nodeSetNodes = []
        nNodeSets = len(exodusDataset.dimensions["num_node_sets"])
        for i in range(nNodeSets):
            key = 'node_ns' + str(i + 1)
            record = exodusDataset.variables[key]
            record.set_auto_mask(False)
            nodeSetNodes.append(record[:] - 1)
        nodeSets = dict(zip(nodeSetNames, nodeSetNodes))
    else:
        nodeSets = {}

    return nodeSets


def _read_side_sets(exodusDataset):
    if "num_side_sets" in exodusDataset.dimensions:
        sideSetNames = _read_names_list(exodusDataset, 'ss_names')
        for i, name in enumerate(sideSetNames):
            if not name:
                sideSetNames[i] = "sideset_" + str(i+1)

        nSideSets = len(exodusDataset.dimensions['num_side_sets'])
        sideSetEntries = []
        for i in range(nSideSets):
            key = 'elem_ss' + str(i + 1)
            record = exodusDataset.variables[key]
            record.set_auto_mask(False)
            sideSetElems = np.array(record[:] - 1)

            key = 'side_ss' + str(i + 1)
            record = exodusDataset.variables[key]
            record.set_auto_mask(False)
            sideSetSides = np.array(record[:] - 1)

            sideSetEntries.append(np.column_stack((sideSetElems, sideSetSides)))
        sideSets = dict(zip(sideSetNames, sideSetEntries))
    else:
        sideSets = {}

    return sideSets


def _read_element_type(exodusDataset):
    elemType = exodusDataset.variables['connect1'].elem_type

    nBlocks = len(exodusDataset.dimensions['num_el_blk'])
    for i in range(nBlocks):
        key = 'connect' + str(i + 1)
        blockElemType = exodusDataset[key].elem_type
        assert (blockElemType == elemType), 'Different element types present in exodus file'
    return elemType


def _read_names_list(exodusDataset, recordName):
    record = exodusDataset.variables[recordName]
    record.set_auto_mask(False)
    namesList = [b"".join(c).decode("UTF-8") for c in record[:]]
    return namesList


def _get_vertex_nodes_from_exodus_tri6_mesh(conns):
    vertexSet = set(conns[:,:3].ravel().tolist())
    vertices = [v for v in vertexSet]
    return np.array(vertices, dtype = np.int_)
