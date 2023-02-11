import numpy as onp
from matplotlib import pyplot as plt

from optimism.JaxConfig import *
from optimism.test import MeshFixture
from optimism import Mesh
from optimism import Interpolants

#from matplotlib import pyplot as plt

class TestSingleMeshFixture(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 2
        xRange = [0.,1.]
        yRange = [0.,1.]

        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        
        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x : self.targetDispGrad.dot(x))


    def test_create_nodesets_from_sidesets(self):
        nodeSets = Mesh.create_nodesets_from_sidesets(self.mesh)

        # this test relies on the fact that matching nodesets
        # and sidesets were created on the MeshFixture
        
        for key in self.mesh.sideSets:
            self.assertArrayEqual(np.sort(self.mesh.nodeSets[key]), nodeSets[key])

    
    def test_edge_connectivities(self):
        edgeConns, _ = Mesh.create_edges(self.mesh.conns)

        goldBoundaryEdgeConns = np.array([[0, 1],
                                          [1, 2],
                                          [2, 5],
                                          [5, 4],
                                          [4, 3],
                                          [3, 0]])

        # Check that every boundary edge has been found.
        # Boundary edges must appear with the same connectivity order,
        # since by convention boundary edge connectivities go
        # in the counter-clockwise sense.

        nBoundaryEdges = goldBoundaryEdgeConns.shape[0]
        boundaryEdgeFound = onp.full(nBoundaryEdges, False)

        for i, be in enumerate(goldBoundaryEdgeConns):
            rowsMatchingGold = onp.all(edgeConns == be, axis=1)
            boundaryEdgeFound[i] = onp.any(rowsMatchingGold)

        self.assertTrue(onp.all(boundaryEdgeFound))

        # Check that every interior edge as been found.
        # Interior edges have no convention defining which
        # sense the vertices should be ordered, so we check
        # for both permutations.

        goldInteriorEdgeConns = np.array([[0, 4],
                                          [1, 4],
                                          [1, 5]])

        nInteriorEdges = goldInteriorEdgeConns.shape[0]
        interiorEdgeFound = onp.full(nInteriorEdges, False)
        for i, e in enumerate(goldInteriorEdgeConns):
            foundWithSameSense = onp.any(onp.all(edgeConns == e, axis=1))
            foundWithOppositeSense = onp.any(onp.all(edgeConns == onp.flip(e), axis=1))
            interiorEdgeFound[i] = foundWithSameSense or foundWithOppositeSense

        self.assertTrue(onp.all(interiorEdgeFound))


    def test_edge_to_neighbor_cells_data(self):
        edgeConns, edges = Mesh.create_edges(self.mesh.conns)

        goldBoundaryEdgeConns = np.array([[0, 1],
                                          [1, 2],
                                          [2, 5],
                                          [5, 4],
                                          [4, 3],
                                          [3, 0]])

        goldBoundaryEdges = onp.array([[0, 0, -1, -1],
                                       [2, 0, -1, -1],
                                       [2, 1, -1, -1],
                                       [3, 1, -1, -1],
                                       [1, 1, -1, -1],
                                       [1, 2, -1, -1]])

        for be, bc in zip(goldBoundaryEdges, goldBoundaryEdgeConns):
            i = np.where(onp.all(edgeConns == bc, axis=1))
            self.assertTrue(np.all(edges[i,:] == be))

        goldInteriorEdgeConns = np.array([[0, 4],
                                          [1, 4],
                                          [5, 1]])
        goldInteriorEdges = onp.array([[1, 0, 0, 2],
                                       [0, 1, 3, 2],
                                       [2, 2, 3, 0]])

        for ie, ic in zip(goldInteriorEdges, goldInteriorEdgeConns):
            foundWithSameSense = onp.any(onp.all(edgeConns == ic, axis=1))
            foundWithOppositeSense = onp.any(onp.all(edgeConns == onp.flip(ic), axis=1))
            edgeDataMatches = False
            if foundWithSameSense:
                i = onp.where(onp.all(edgeConns == ic, axis=1))
                edgeData = ie
            elif foundWithOppositeSense:
                i = onp.where((onp.all(edgeConns == onp.flip(ic), axis=1)))
                edgeData = ie[[2, 3, 0, 1]]
            else:
                self.fail('edge not found with vertices ' + str(ic))
            edgeDataMatches = np.all(edges[i,:] == edgeData)
            self.assertTrue(edgeDataMatches)


    def test_conversion_to_quadratic_mesh_is_valid(self):
        newMesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, 2)

        nNodes = newMesh.coords.shape[0]
        self.assertEqual(nNodes, 15)

        # make sure all of the newly created nodes got used in the connectivity
        self.assertArrayEqual(np.unique(newMesh.conns.ravel()), np.arange(nNodes))

        # check that all triangles are valid:
        # compute inradius of each triangle and of the sub-triangle of the mid-edge nodes
        # Both should be nonzero, and parent inradius should be 2x sub-triangle inradius
        master = newMesh.parentElement
        for t in newMesh.conns:
            elCoords = newMesh.coords[t, :]
            parentCoords = elCoords[master.vertexNodes, :]
            midEdgeNodes = master.faceNodes[:, 1]
            childCoords = elCoords[midEdgeNodes, :]

            parentArea = triangle_inradius(parentCoords)
            childArea = triangle_inradius(childCoords)
            
            self.assertGreater(parentArea, 0.0)
            self.assertGreater(childArea, 0.0)
            self.assertAlmostEqual(parentArea, 2.0*childArea, 10)

        # uncomment the following to inspect higher order mesh
        # print('coords=\n', newMesh.coords)
        # print('conns=\n', newMesh.conns)
        # plt.triplot(newMesh.coords[:,0], newMesh.coords[:,1], newMesh.conns[:,master.vertexNodes])
        # plt.scatter(newMesh.coords[:,0], newMesh.coords[:,1], marker='o')
        # plt.show()


def triangle_inradius(tcoords):
    area = 0.5*onp.cross(tcoords[1]-tcoords[0], tcoords[2]-tcoords[0])
    peri = (onp.linalg.norm(tcoords[1]-tcoords[0])
            + onp.linalg.norm(tcoords[2]-tcoords[1])
            + onp.linalg.norm(tcoords[0]-tcoords[2]))
    return area/peri

if __name__ == '__main__':
    MeshFixture.unittest.main()

