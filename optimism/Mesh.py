import numpy as onp
from optimism.JaxConfig import *
from optimism import Interpolants


Mesh = namedtuple('Mesh', ['coords','conns','simplexNodesOrdinals',
                           'parentElement', 'parentElement1d', 'blocks',
                           'nodeSets', 'sideSets'],
                  defaults=(None,None,None))
Mesh.__doc__ = \
    """Triangle mesh representing a domain.

    Attributes:
        coords: Coordinates of the nodes, shape ``(nNodes, nDim)``.
        conns: Nodal connectivity table of the elements.
        simplexNodesOrdinals: Indices of the nodes that are vertices.
        parentElement: A ``ParentElement`` that is the element type in
            parametric space. A mesh can contain only 1 element type.
        parentElement1d:
        blocks: A dictionary mapping element block names to the indices of the
            elements in the block.
        nodeSets: A dictionary mapping node set names to the indices of the
            nodes.
        sideSets: A dictionary mapping side set names to the edges. The
            edge data structure is a tuple of the element index and the local
            number of the edge within that element. For example, triangle
            elements will have edge 0, 1, or 2 for this entry.
    """


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
    element, element1d = Interpolants.make_parent_elements(degree=1)
    vertexNodes = np.arange(coords.shape[0])
    return Mesh(coords, conns, vertexNodes,
                element, element1d, blocks, nodeSets, sideSets)


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
    assert mesh1.parentElement.degree == 1
    assert mesh2.parentElement.degree == 1

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
                mesh1.parentElement, mesh1.parentElement1d, blocks,
                nodeSets, sideSets), disp


def mesh_with_coords(mesh, coords):
    return Mesh(coords, mesh.conns,mesh.simplexNodesOrdinals,
                mesh.parentElement, mesh.parentElement1d, mesh.blocks, mesh.nodeSets, mesh.sideSets)


def mesh_with_nodesets(mesh, nodeSets):
    return Mesh(mesh.coords, mesh.conns,
                mesh.simplexNodesOrdinals,
                mesh.parentElement, mesh.parentElement1d,
                mesh.blocks, nodeSets, mesh.sideSets)


def mesh_with_blocks(mesh, blocks):
    return Mesh(mesh.coords, mesh.conns,
                mesh.simplexNodesOrdinals,
                mesh.parentElement, mesh.parentElement1d,
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

    parentElement1d = Interpolants.make_parent_element_1d(order)
    
    if useBubbleElement:
        basis = Interpolants.make_parent_element_2d_with_bubble(order)
    else:
        basis = Interpolants.make_parent_element_2d(order)

    conns = np.zeros((num_elements(mesh), basis.coordinates.shape[0]), dtype=np.int_)

    # step 1/3: vertex nodes
    conns = conns.at[:,basis.vertexNodes].set(mesh.conns)
    simplexNodesOrdinals = np.arange(mesh.coords.shape[0])

    nodeOrdinalOffset = mesh.coords.shape[0] # offset for later node numbering

    # step 2/3: mid-edge nodes (excluding vertices)
    edgeConns, edges = create_edges(mesh.conns)
    A = np.column_stack((1.0-parentElement1d.coordinates[parentElement1d.interiorNodes],
                         parentElement1d.coordinates[parentElement1d.interiorNodes]))
    edgeCoords = vmap(lambda edgeConn: np.dot(A, mesh.coords[edgeConn,:]))(edgeConns)

    nNodesPerEdge = parentElement1d.interiorNodes.size
    for e, edge in enumerate(edges):
        edgeNodeOrdinals = nodeOrdinalOffset + np.arange(e*nNodesPerEdge,(e+1)*nNodesPerEdge)
        
        elemLeft = edge[0]
        sideLeft = edge[1]
        edgeMasterNodes = basis.faceNodes[sideLeft][parentElement1d.interiorNodes]
        conns = conns.at[elemLeft, edgeMasterNodes].set(edgeNodeOrdinals)

        elemRight = edge[2]
        if elemRight >= 0:
            sideRight = edge[3]
            edgeMasterNodes = basis.faceNodes[sideRight][parentElement1d.interiorNodes]
            conns = conns.at[elemRight, edgeMasterNodes].set(np.flip(edgeNodeOrdinals))

    nEdges = edges.shape[0]
    nodeOrdinalOffset += nEdges*nNodesPerEdge # for offset of interior node numbering

    # step 3/3: interior nodes
    nInNodesPerTri = basis.interiorNodes.shape[0]
    if nInNodesPerTri > 0:
        N0 = basis.coordinates[basis.interiorNodes,0]
        N1 = basis.coordinates[basis.interiorNodes,1]
        N2 = 1.0 - N0 - N1
        A = np.column_stack((N0,N1,N2))
        interiorCoords = vmap(lambda triConn: np.dot(A, mesh.coords[triConn]))(mesh.conns)

        def add_element_interior_nodes(conn, newNodeOrdinals):
            return conn.at[basis.interiorNodes].set(newNodeOrdinals)

        nTri = conns.shape[0]
        newNodeOrdinals = np.arange(nTri*nInNodesPerTri).reshape(nTri,nInNodesPerTri) \
            + nodeOrdinalOffset
        
        conns = vmap(add_element_interior_nodes)(conns, newNodeOrdinals)
    else:
        interiorCoords = np.zeros((0,2))


    coords = np.vstack((mesh.coords,edgeCoords.reshape(-1,2),interiorCoords.reshape(-1,2)))
    nodeSets = mesh.nodeSets if copyNodeSets else None

    newMesh = Mesh(coords, conns, simplexNodesOrdinals, basis,
                   parentElement1d, mesh.blocks, nodeSets, mesh.sideSets)
    
    if createNodeSetsFromSideSets:
        nodeSets = create_nodesets_from_sidesets(newMesh)
        newMesh = mesh_with_nodesets(newMesh, nodeSets)

    return newMesh


def create_nodesets_from_sidesets(mesh):
    nodeSets = {}

    def get_nodes_from_edge(edge):
        elemOrdinal = edge[0]
        sideOrdinal = edge[1]
        return mesh.conns[elemOrdinal, mesh.parentElement.faceNodes[sideOrdinal,:]]
    
    for setName, sideSet in mesh.sideSets.items():
        nodes = vmap(get_nodes_from_edge)(sideSet)
        nodeSets[setName] = np.unique(nodes.ravel())

    return nodeSets


def num_elements(mesh):
    return mesh.conns.shape[0]


def num_nodes(mesh):
    return mesh.coords.shape[0]


def get_edge_coords(mesh, edge):
    """Get coordinates of nodes on an element edge.

    Arguments:
    mesh: a Mesh object
    edge: tuple containing the element number containing the edge and the
        permutation (0, 1, or 2) of the edge within the triangle
    """
    edgeNodes = mesh.parentElement.faceNodes[edge[1], :]
    nodes = mesh.conns[edge[0], edgeNodes]
    return mesh.coords[nodes]


def compute_edge_vectors(mesh, edgeCoords):
    """Get geometric vectors for an element edge.
    
    Assumes that the edgs has a constant shape jacobian, that is, the
    transformation from the parent element is affine.
    
    Arguments:
    mesh: a Mesh object
    edgeCoords: coordinates of all nodes on the edge, in the order
        defined by the 1D parent element convention

    Returns:
    tuple (t, n, j) with
    t: the unit tangent vector
    n: the outward unit normal vector
    j: jacobian of the transformation from parent to physical space
    """
    Xv = edgeCoords[mesh.parentElement1d.vertexNodes, :]
    tangent = Xv[1] - Xv[0]
    normal = np.array([tangent[1], -tangent[0]])
    jac = np.linalg.norm(tangent)
    return tangent/jac, normal/jac, jac