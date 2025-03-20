from optimism.JaxConfig import *
from optimism import Mesh
from optimism.test.MeshFixture import MeshFixture
from optimism.contact import MortarContact
import unittest

class TwoBodyContactFixture(MeshFixture):

    def setUp(self):
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])
        
        m1 = self.create_mesh_and_disp(3, 5, [0.0, 1.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '1')

        m2 = self.create_mesh_and_disp(2, 8, [0.9, 2.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '2')
        
        self.mesh, _ = Mesh.combine_mesh(m1, m2)
        order=2
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=order, copyNodeSets=False)
        self.disp = np.zeros(self.mesh.coords.shape)

        sideA = self.mesh.sideSets['right1']
        sideB = self.mesh.sideSets['left2']
        
        self.segmentConnsA = MortarContact.get_facet_connectivities(self.mesh, sideA)
        self.segmentConnsB = MortarContact.get_facet_connectivities(self.mesh, sideB)


    def plot_solution(self, plotName):
        from optimism import VTKWriter
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=self.disp,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        writer.add_contact_edges(self.segmentConnsA)
        writer.add_contact_edges(self.segmentConnsB)
        writer.write()
        

    @unittest.skipIf(False, '')
    def test_contact_search(self):
        #self.plot_solution('mesh')

        neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, 5)

        def compute_overlap_area(segB, neighborSegsA):
            def compute_area_for_segment_pair(segB, indexA):
                segA = self.segmentConnsA[indexA]
                coordsSegB = self.mesh.coords[segB] + self.disp[segB]
                coordsSegA = self.mesh.coords[segA] + self.disp[segA]
                return MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: 1.0, 1e-9, f_clamp_gaps=MortarContact.no_clamp)
            
            return np.sum(vmap(compute_area_for_segment_pair, (None,0))(segB, neighborSegsA))
        
        totalSum = np.sum(vmap(compute_overlap_area)(self.segmentConnsB, neighborList))
        self.assertNear(totalSum, 1.0, 5)


    def test_contact_constraints(self):
        #self.plot_solution('mesh')

        neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, 5)

        nodalGapField = MortarContact.assemble_area_weighted_gaps(self.mesh.coords, self.disp, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal)
        nodalAreaField = MortarContact.assemble_nodal_areas(self.mesh.coords, self.disp, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal)

        nodalGapField = vmap(lambda x, d : np.where( d > 0, x / d, 0.0))(nodalGapField, nodalAreaField)

        mask = np.ones(len(nodalAreaField), dtype=np.int8)
        nodesB = np.unique(np.concatenate(self.segmentConnsB))
        mask = mask.at[nodesB].set(0).astype(bool)

        self.assertNear(np.sum(nodalAreaField), 1.0, 8)
        self.assertNear(np.sum(nodalAreaField), np.sum(nodalAreaField[nodesB]), 14)
        self.assertNear(np.sum(nodalAreaField[mask]), 0.0, 14)


class TwoBodySelfContactFixture(MeshFixture):

    def setUp(self):
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])

        m1 = self.create_mesh_and_disp(2, 4, [0.0, 1.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '1')

        # Have second block offest (not penetrating) to test clamping of gaps for values larger than half the block width.
        # Clamping removes gap values computed for edges on opposite faces of a body (they shouldn't be considered as penetrating)
        m2 = self.create_mesh_and_disp(2, 3, [1.1, 2.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '2')
        
        self.mesh, _ = Mesh.combine_mesh(m1, m2)
        order=2
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=order, copyNodeSets=False)
        self.disp = np.zeros(self.mesh.coords.shape)

        self.numNeighbors = 8 # include enough neighbors so that search includes edges on opposite faces

        # combine all edges in same sideset to test "self-contact" (even though it's separate blocks)
        combinedSideSets = np.unique(np.concatenate(tuple(self.mesh.sideSets.values()), axis=0), axis=0)
        self.segmentConnsA = MortarContact.get_facet_connectivities(self.mesh, combinedSideSets)
        self.segmentConnsB = self.segmentConnsA
    

    def test_neighbor_search_excludes_current_and_adjacent_edges(self):
        neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, self.numNeighbors)

        def edges_are_adjacent(edge1, edge2):
            return (edge1[0] == edge2[0]) | \
                   (edge1[0] == edge2[1]) | \
                   (edge1[1] == edge2[0]) | \
                   (edge1[1] == edge2[1])

        self.assertEqual(neighborList.shape[0], self.segmentConnsA.shape[0])

        for i in range(neighborList.shape[0]):
            currentEdge = self.segmentConnsA[i]
            for neighborIdx in neighborList[i]:
                neighborEdge = self.segmentConnsA[neighborIdx]
                self.assertFalse(edges_are_adjacent(currentEdge, neighborEdge))


    def test_mortar_integral_computes_expected_area(self):
        nodesA = np.unique(np.concatenate(self.segmentConnsA))
        nodesB = np.unique(np.concatenate(self.segmentConnsB))
        self.assertArrayEqual(nodesA, nodesB)

        neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, self.numNeighbors)

        nodalAreaField = MortarContact.assembly_mortar_integral(self.mesh.coords, 
                                                                self.disp, 
                                                                self.segmentConnsA, 
                                                                self.segmentConnsB, 
                                                                neighborList, 
                                                                MortarContact.compute_average_normal, 
                                                                lambda gap: 1.0, 
                                                                MortarContact.no_clamp)
        clampedNodalAreaField = MortarContact.assembly_mortar_integral(self.mesh.coords, 
                                                                       self.disp, 
                                                                       self.segmentConnsA, 
                                                                       self.segmentConnsB, 
                                                                       neighborList, 
                                                                       MortarContact.compute_average_normal, 
                                                                       lambda gap: 1.0, 
                                                                       lambda xiA, g: MortarContact.clamp_max_penetration(xiA, g, max_penetration_depth=0.5))

        # check that area is wrong if clamping is not used (gaps computed for edges on opposite faces)
        self.assertNotAlmostEqual(np.sum(nodalAreaField), 2.0, 8) # double counting each edge, so area is doubled

        # check that area is correct if clamping is used 
        self.assertNear(np.sum(clampedNodalAreaField), 2.0, 8) # double counting each edge, so area is doubled
        self.assertNear(np.sum(clampedNodalAreaField), np.sum(clampedNodalAreaField[nodesA]), 14)
        mask = np.ones(len(clampedNodalAreaField), dtype=np.int8)
        mask = mask.at[nodesA].set(0).astype(bool)

        self.assertNear(np.sum(clampedNodalAreaField[mask]), 0.0, 14)


    def test_clamped_mortar_integral_computes_no_penetration(self):
        neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, self.numNeighbors)
        nodalGapField = MortarContact.assemble_area_weighted_gaps(self.mesh.coords, self.disp, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal)
        clampedNodalGapField = MortarContact.assemble_area_weighted_clamped_gaps(self.mesh.coords, self.disp, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal, lambda xiA, g: MortarContact.clamp_max_penetration(xiA, g, max_penetration_depth=0.5))

        self.assertFalse(np.all(nodalGapField >= 0.0)) # negative gap values computed since edges on opposite faces are included
        self.assertTrue(np.all(clampedNodalGapField >= 0.0)) # negative gap values computed since edges on opposite faces are included


if __name__ == '__main__':
    unittest.main()

