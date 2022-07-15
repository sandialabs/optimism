import numpy as onp
from optimism.JaxConfig import *
from optimism import Interpolants
from optimism import TensorMath


Mesh = namedtuple('Mesh', ['coords','conns','simplexNodesOrdinals',
                           'masterElement', 'masterLineElement', 'blocks',
                           'nodeSets', 'sideSets'],
                  defaults=(None,None,None))

EssentialBC = namedtuple('EssentialBC', ['nodeSet', 'field'])


def create_structured_mesh_data(Nx, Ny, xExtent, yExtent):
    xs = np.linspace(xExtent[0], xExtent[1], Nx)
    ys = np.linspace(yExtent[0], yExtent[1], Ny)

    Ex = Nx-1
    Ey = Ny-1

    coords = [ [xs[nx], ys[ny]] for ny in range(Ny) for nx in range(Nx) ]

    conns = []
    for ex in range(Ex):
        for ey in range(Ey):
            conns.append([ex + Nx*ey, ex+1 + Nx*ey,     ex+1 + Nx*(ey+1)])
            conns.append([ex + Nx*ey, ex+1 + Nx*(ey+1), ex   + Nx*(ey+1)])

    coords = np.array(coords)
    conns = np.array(conns)
    return (coords, conns)


def construct_mesh_from_basic_data(coords, conns, blocks, nodeSets=None, sideSets=None):
    master, master1d = Interpolants.make_master_elements(degree=1)
    vertexNodes = np.arange(coords.shape[0])
    return Mesh(coords, conns, vertexNodes,
                master, master1d, blocks, nodeSets, sideSets)


def construct_structured_mesh(Nx, Ny, xExtent, yExtent, elementOrder=1, useBubbleElement=False):
    coords, conns = create_structured_mesh_data(Nx, Ny, xExtent, yExtent)
    blocks = {'block_0': np.arange(conns.shape[0])}
    mesh = construct_mesh_from_basic_data(coords, conns, blocks)
    if elementOrder > 1:
        mesh = create_higher_order_mesh_from_simplex_mesh(mesh, elementOrder, useBubbleElement)
    return mesh


def get_blocks(mesh, blockNames):
    return tuple(mesh.blocks[name] for name in blockNames)

    
def combine_nodesets(set1, set2, nodeOffset):
    newSet = {}
    if set1!=None:
        for key in set1:
            newSet[key] = set1[key]
    if set2!=None:
        for key in set2:
            newSet[key] = set2[key] + nodeOffset
    return newSet


def combine_sidesets(set1, set2, elemOffset):
    newSet = {}
    if set1!=None:
        for key in set1:
            val = set1[key]
            newSet[key] = val
    if set2!=None:
        for key in set2:
            val = set2[key]
            newSet[key] = val.at[:,0].add(elemOffset) if len(val)>0 else np.array([])
    return newSet


def combine_blocks(set1, set2, elemOffset):
    # meshes are required to have at least one block
    newSet = {}

    for key in set1:
        val = set1[key]
        newSet[key] = val

    for key in set2:
        val = set2[key]
        newSet[key] = val + elemOffset
    return newSet


def combine_mesh(m1, m2):
    # need to implement block combining
    
    mesh1,disp1 = m1
    mesh2,disp2 = m2

    # Only handle 2 linear element meshes
    assert mesh1.masterElement.degree == 1
    assert mesh2.masterElement.degree == 1

    numNodes1 = mesh1.coords.shape[0]
    numElems1 = num_elements(mesh1)

    coords = np.concatenate((mesh1.coords, mesh2.coords), axis=0)
    conns = np.concatenate((mesh1.conns, mesh2.conns+numNodes1), axis=0)
    disp = np.concatenate((disp1, disp2), axis=0)

    nodeSets = None
    if mesh1.nodeSets!=None or mesh2.nodeSets!=None:
        nodeSets = combine_nodesets(mesh1.nodeSets, mesh2.nodeSets, numNodes1)

    sideSets = None
    if mesh1.sideSets!=None or mesh2.sideSets!=None:
        sideSets = combine_sidesets(mesh1.sideSets, mesh2.sideSets, numElems1)

    simplexNodeOrdinals = np.arange(coords.shape[0])

    blocks = combine_blocks(mesh1.blocks, mesh2.blocks, numElems1)
        
    return Mesh(coords, conns, simplexNodeOrdinals,
                mesh1.masterElement, mesh1.masterLineElement, blocks,
                nodeSets, sideSets), disp


def mesh_with_coords(mesh, coords):
    return Mesh(coords, mesh.conns,mesh.simplexNodesOrdinals,
                mesh.masterElement, mesh.masterLineElement, mesh.blocks, mesh.nodeSets, mesh.sideSets)


def mesh_with_nodesets(mesh, nodeSets):
    return Mesh(mesh.coords, mesh.conns,
                mesh.simplexNodesOrdinals,
                mesh.masterElement, mesh.masterLineElement,
                mesh.blocks, nodeSets, mesh.sideSets)


def mesh_with_blocks(mesh, blocks):
    return Mesh(mesh.coords, mesh.conns,
                mesh.simplexNodesOrdinals,
                mesh.masterElement, mesh.masterLineElement,
                blocks, mesh.nodeSets, mesh.sideSets)

def create_edges(conns):
    """Generate topological information about edges in a triangulation.

    Parameters
    ----------
    conns : (nTriangles, 3) array
        Connectivity table of the triangulation.

    Returns
    -------
    edgeConns : (nEdges, 2) array
        Vertices of each edge. Boundary edges are always in the
        counter-clockwise sense, so that the interior of the body is on the left
        side when walking from the first vertex to the second.
    edges : (nEdges, 4) array
        Edge-to-triangle topological information. Each row provides the
        follwing information for each edge: [leftT, leftP, rightT, rightP],
        where leftT is the ID of the triangle to the left, leftP is the
        permutation of the edge in the left triangle (edge 0, 1, or 2), rightT
        is the ID of the triangle to the right, and rightP is the permutation
        of the edge in the right triangle. If the edge is a boundary edge, the
        values of rightT and rightP are -1.
    """
    nTris = conns.shape[0]
    allTriFaces = onp.vstack((conns[:, (0,1)], conns[:, (1,2)], conns[:, (2,0)]))
    foo = onp.sort(allTriFaces, axis=1)
    bar, i = onp.unique(foo, return_index=True, axis=0)
    edgeConns = (allTriFaces[i,:])

    nEdges = edgeConns.shape[0]
    edges = -onp.ones((nEdges, 4), dtype=onp.int_)
    edgeElementIds = onp.tile(np.arange(nTris), 3)
    edges[:,0] = edgeElementIds[i]
    edges[:,1] = i // nTris

    for i, ec in enumerate(edgeConns):
        rowsMatch = onp.all(onp.flip(ec) == allTriFaces, axis=1)
        if onp.any(rowsMatch):
            j = onp.where(rowsMatch)[0]
            # there should only be one matching row, but take element 0
            # because j will have the same number of axes (2) as
            # rowsMatch.
            edges[i, 2] = edgeElementIds[j]
            edges[i, 3] = j // nTris

    return edgeConns, edges


def create_higher_order_mesh_from_simplex_mesh(mesh, order, useBubbleElement=False, copyNodeSets=False, createNodeSetsFromSideSets=False):
    if order==1: return mesh

    master1d = Interpolants.make_master_line_element(order)
    
    if useBubbleElement:
        master = Interpolants.make_master_tri_bubble_element(order)
    else:
        master = Interpolants.make_master_tri_element(order)

    conns = np.zeros((num_elements(mesh), master.coordinates.shape[0]), dtype=np.int_)

    # step 1/3: vertex nodes
    conns = conns.at[:,master.vertexNodes].set(mesh.conns)
    simplexNodesOrdinals = np.arange(mesh.coords.shape[0])

    nodeOrdinalOffset = mesh.coords.shape[0] # offset for later node numbering

    # step 2/3: mid-edge nodes (excluding vertices)
    edgeConns, edges = create_edges(mesh.conns)
    A = np.column_stack((1.0-master1d.coordinates[master1d.interiorNodes],
                         master1d.coordinates[master1d.interiorNodes]))
    edgeCoords = vmap(lambda edgeConn: np.dot(A, mesh.coords[edgeConn,:]))(edgeConns)

    nNodesPerEdge = master1d.interiorNodes.size
    for e, edge in enumerate(edges):
        edgeNodeOrdinals = nodeOrdinalOffset + np.arange(e*nNodesPerEdge,(e+1)*nNodesPerEdge)
        
        elemLeft = edge[0]
        sideLeft = edge[1]
        edgeMasterNodes = master.faceNodes[sideLeft][master1d.interiorNodes]
        conns = conns.at[elemLeft, edgeMasterNodes].set(edgeNodeOrdinals)

        elemRight = edge[2]
        if elemRight >= 0:
            sideRight = edge[3]
            edgeMasterNodes = master.faceNodes[sideRight][master1d.interiorNodes]
            conns = conns.at[elemRight, edgeMasterNodes].set(np.flip(edgeNodeOrdinals))

    nEdges = edges.shape[0]
    nodeOrdinalOffset += nEdges*nNodesPerEdge # for offset of interior node numbering

    # step 3/3: interior nodes
    nInNodesPerTri = master.interiorNodes.shape[0]
    if nInNodesPerTri > 0:
        N0 = master.coordinates[master.interiorNodes,0]
        N1 = master.coordinates[master.interiorNodes,1]
        N2 = 1.0 - N0 - N1
        A = np.column_stack((N0,N1,N2))
        interiorCoords = vmap(lambda triConn: np.dot(A, mesh.coords[triConn]))(mesh.conns)

        def add_element_interior_nodes(conn, newNodeOrdinals):
            return conn.at[master.interiorNodes].set(newNodeOrdinals)

        nTri = conns.shape[0]
        newNodeOrdinals = np.arange(nTri*nInNodesPerTri).reshape(nTri,nInNodesPerTri) \
            + nodeOrdinalOffset
        
        conns = vmap(add_element_interior_nodes)(conns, newNodeOrdinals)
    else:
        interiorCoords = np.zeros((0,2))


    coords = np.vstack((mesh.coords,edgeCoords.reshape(-1,2),interiorCoords.reshape(-1,2)))
    nodeSets = mesh.nodeSets if copyNodeSets else None

    newMesh = Mesh(coords, conns, simplexNodesOrdinals, master,
                   master1d, mesh.blocks, nodeSets, mesh.sideSets)
    
    if createNodeSetsFromSideSets:
        nodeSets = create_nodesets_from_sidesets(newMesh)
        newMesh = mesh_with_nodesets(newMesh, nodeSets)

    return newMesh


def create_nodesets_from_sidesets(mesh):
    master = mesh.masterElement
    nodeSets = {}

    def get_nodes_from_edge(edge):
        elemOrdinal = edge[0]
        sideOrdinal = edge[1]
        return mesh.conns[elemOrdinal,master.faceNodes[sideOrdinal,:]]
    
    for setName, sideSet in mesh.sideSets.items():
        nodes = vmap(get_nodes_from_edge)(sideSet)
        nodeSets[setName] = np.unique(nodes.ravel())

    return nodeSets


def num_elements(mesh):
    return mesh.conns.shape[0]


def num_nodes(mesh):
    return mesh.coords.shape[0]


class DofManager:
    
    def __init__(self, mesh, fieldShape, EssentialBCs=[]):
        self.fieldShape = fieldShape
        self.isBc = onp.full(fieldShape, False, dtype=bool)
        for ebc in EssentialBCs:
            self.isBc[mesh.nodeSets[ebc.nodeSet], ebc.field] = True
        self.isUnknown = ~self.isBc
        
        self.ids = np.arange(self.isBc.size).reshape(fieldShape)
        
        self.unknownIndices = self.ids[self.isUnknown]
        self.bcIndices = self.ids[self.isBc]
        
        ones = np.ones(self.isBc.size, dtype=int) * -1
        self.dofToUnknown = ones.at[self.unknownIndices].set(np.arange(self.unknownIndices.size)) 

        self.HessRowCoords, self.HessColCoords = self._make_hessian_coordinates(mesh.conns)

        self.hessian_bc_mask = self._make_hessian_bc_mask(mesh.conns)
        
        
    def get_bc_size(self):
        return np.sum(self.isBc)


    def get_unknown_size(self):
        return np.sum(self.isUnknown)
    
    
    def create_field(self, Uu, Ubc=0.0):
        U = np.zeros(self.isBc.shape).at[self.isBc].set(Ubc)
        return U.at[self.isUnknown].set(Uu)
    

    def get_bc_values(self, U):
        return U[self.isBc]


    def get_unknown_values(self, U):
        return U[self.isUnknown]

    
    def slice_unknowns_with_dof_indices(self, Uu, dofIndexSlice):
        i = self.isUnknown[dofIndexSlice]
        j = self.dofToUnknown.reshape(self.fieldShape)[dofIndexSlice]
        return Uu[j[i]]


    def _make_hessian_coordinates(self, conns):
        nElUnknowns = onp.zeros(conns.shape[0], dtype=int)
        nHessianEntries = 0
        for e, eNodes in enumerate(conns):
            elUnknownFlags = self.isUnknown[eNodes,:].ravel()
            nElUnknowns[e] = onp.sum(elUnknownFlags)
            nHessianEntries += onp.square(nElUnknowns[e])
       
        rowCoords = onp.zeros(nHessianEntries, dtype=int)
        colCoords = rowCoords.copy()
        rangeBegin = 0
        for e,eNodes in enumerate(conns):
            elDofs = self.ids[eNodes,:]
            elUnknownFlags = self.isUnknown[eNodes,:]
            elUnknowns = self.dofToUnknown[elDofs[elUnknownFlags]]
            elHessCoords = onp.tile(elUnknowns, (nElUnknowns[e],1))

            rangeEnd = rangeBegin + onp.square(nElUnknowns[e])

            rowCoords[rangeBegin:rangeEnd] = elHessCoords.ravel()
            colCoords[rangeBegin:rangeEnd] = elHessCoords.T.ravel()

            rangeBegin += np.square(nElUnknowns[e])
        return rowCoords, colCoords


    def _make_hessian_bc_mask(self, conns):
        nElements, nNodesPerElement = conns.shape
        nFields = self.ids.shape[1]
        nDofPerElement = nNodesPerElement*nFields

        hessian_bc_mask = onp.full((nElements,nDofPerElement,nDofPerElement),
                                   True, dtype=bool)
        for e, eNodes in enumerate(conns):
            eFlag = self.isBc[eNodes,:].ravel()
            hessian_bc_mask[e,eFlag,:] = False
            hessian_bc_mask[e,:,eFlag] = False
        return hessian_bc_mask

