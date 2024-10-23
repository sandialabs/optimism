from optimism.JaxConfig import *
from optimism import Mesh
from optimism import Interpolants
import netCDF4
import numpy as onp

exodusToNativeTri6NodeOrder = np.array([0, 3, 1, 5, 4, 2])


def read__mesh(fileName):
    with netCDF4.Dataset(fileName) as exData:
        coords = _read_coordinates(exData)
        conns, blocks = _read_blocks(exData)
        nodeSets = _read_node_sets(exData)
        sideSets = _read_side_sets(exData)
        connsn = onp.array(conns)


        # Rearranging arrays; we are only reading one block so stuff should only come from there.
        # Get foreground Node indices first.
        FG_Node_Indices = onp.transpose(onp.unique(onp.reshape(connsn,-1)))
        
        # Now, get the coordinates corresponding to that
        coords_FG = onp.array(coords[FG_Node_Indices,:])

        
        
        # Now, overwrite conns, because that still uses the old indices
        for i in range(onp.shape(connsn)[0]):
            for j in range(onp.shape(connsn)[1]):
                val = connsn[i,j]
                ind = onp.where(FG_Node_Indices == val)
                indv = ind[0].item()
                connsn[i,j] = indv
        
        conns = np.array(connsn)

        # Finally, overwrite coords, so that all the other stuff can come from it.
        coords = np.array(coords_FG)

        elementType = _read_element_type(exData).lower()
        if elementType == "tri3" or elementType == "tri":
            basis, basis1d = Interpolants.make_parent_elements(degree = 1)
            simplexNodesOrdinals = np.arange(coords.shape[0])
        elif elementType == "tri6":
            basis, basis1d = Interpolants.make_parent_elements(degree = 2)
            simplexNodesOrdinals = _get_vertex_nodes_from_exodus_tri6_mesh(conns)
            conns = conns[:, exodusToNativeTri6NodeOrder]
        #else:
        #    raise Exception('Cannot work with higher than 2nd order elements.')
        
     
        return Mesh.Mesh(coords, conns, simplexNodesOrdinals, basis, basis1d,
                         blocks, nodeSets, sideSets)
        

def read_exodus_mesh_element_properties(fileName, varNames, blockNum=1):
    varValues = []
    with netCDF4.Dataset(fileName) as exData:
        blockNames = _read_names_list(exData, 'eb_names')
        if len(blockNames) > 1:
            raise ValueError('Only single blocks are supported currently')

        for name in varNames:
            varValues.append(_read_block_variable_values(exData, blockNum - 1, name))

    return np.vstack(varValues).T


def _read_coordinates(exodusDataset):
    nNodes = len(exodusDataset.dimensions['num_nodes'])
    nDims = len(exodusDataset.dimensions['num_dim'])
    assert nDims == 2

    coordsX = exodusDataset.variables['coordx'][:]
    coordsY = exodusDataset.variables['coordy'][:]
    
    return np.column_stack([coordsX.filled(), coordsY.filled()])


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
    for i in range(1):
        nodesPerElemInBlock = len(exodusDataset.dimensions['num_nod_per_el' + str(i+1)])
        assert nodesPerElemInBlock == nodesPerElem

        blockConns.append(_read_block_conns(exodusDataset, i))
        
        nElemsInBlock = len(exodusDataset.dimensions['num_el_in_blk' + str(i + 1)])
        elemRange = np.arange(firstElemInBlock, firstElemInBlock + nElemsInBlock)
        blocks[blockNames[i]] = elemRange
        firstElemInBlock += nElemsInBlock

    conns = np.vstack(blockConns)
    return conns, blocks


def _read_block_variable_values(exodusDataset, blockOrdinal, variableName):
    # read element variables currently on mesh
    record = exodusDataset.variables['name_elem_var']
    record.set_auto_mask(False)

    propNamesInMesh = []
    for n in range(record.shape[0]):
        propNamesInMesh.append(''.join(onp.char.decode(record[n, :])))

    # make sure the requested variable is there
    try:
        idx = propNamesInMesh.index(variableName)
    except:
        string = f'Requested variable {variableName} not found on mesh.\n'
        string += 'Available variables are:\n'
        for name in propNamesInMesh:
            string += f'  {name}\n'
        print(string)
        raise KeyError(string)

    # get the name correct
    key = f'vals_elem_var{idx + 1}eb{blockOrdinal + 1}'

    # finally read the variable
    record = exodusDataset.variables[key]
    record.set_auto_mask(False)
    return np.array(record[:])


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
