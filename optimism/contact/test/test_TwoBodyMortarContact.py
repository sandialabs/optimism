from optimism.JaxConfig import *
from optimism import Mesh
from optimism import QuadratureRule
from optimism import Surface
from optimism.contact import EdgeIntersection
from optimism import FunctionSpace
from optimism.test.MeshFixture import MeshFixture
#from optimism.contact import Contact
from optimism.contact import MortarContact
import unittest


#@partial(jit, static_argnums=(4,))
def get_closest_neighbors(edgeSetA : np.array,
                          edgeSetB : np.array,
                          mesh : Mesh.Mesh,
                          disp : np.array,
                          maxNeighbors : int):
    def min_dist_squared(edge1, edge2, coords, disp):
        xs1 = coords[edge1] + disp[edge1]
        xs2 = coords[edge2] + disp[edge2]
        dists = vmap( lambda x: vmap( lambda y: (x-y)@(x-y) )(xs1) ) (xs2)    
        return np.min(dists)
    
    def get_close_edge_indices(surfaceA, edgeB):
        minDistsToA = vmap(min_dist_squared, (0,None,None,None))(surfaceA, edgeB, mesh.coords, disp)
        return np.argsort(minDistsToA)[:maxNeighbors]
    
    return vmap(get_close_edge_indices, (None,0))(edgeSetA, edgeSetB) # loop over surface B, get neighbor index in A


@jit
def get_facet_connectivities(mesh : Mesh.Mesh, sideset):
    def get_sub_segments(side):
        indices = Mesh.get_edge_node_indices(mesh, side)
        return vmap(lambda x,y: np.array([x,y]))(indices[:-1], indices[1:])

    segmentConns = vmap(get_sub_segments)(sideset)
    return segmentConns.reshape((segmentConns.shape[0]*segmentConns.shape[1], segmentConns.shape[2]))


class TwoBodyContactFixture(MeshFixture):

    def setUp(self):
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])
        
        m1 = self.create_mesh_and_disp(3, 4, [0.0, 1.0], [0.0, 0.8],
                                        lambda x : self.targetDispGrad.dot(x), '1')

        m2 = self.create_mesh_and_disp(2, 8, [0.9, 2.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '2')
        
        self.mesh, _ = Mesh.combine_mesh(m1, m2)
        order=2
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=order, copyNodeSets=False)
        self.disp = np.zeros(self.mesh.coords.shape)

        sideA = self.mesh.sideSets['right1']
        sideB = self.mesh.sideSets['left2']
        
        self.segmentConnsA = get_facet_connectivities(self.mesh, sideA)
        self.segmentConnsB = get_facet_connectivities(self.mesh, sideB)
        
    def plot_solution(self, plotName):
        from optimism import VTKWriter

        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='displacement',
                               nodalData=self.disp,
                               fieldType=VTKWriter.VTKFieldType.VECTORS)
        writer.add_contact_edges(self.segmentConnsA)
        writer.add_contact_edges(self.segmentConnsB)
        
        writer.write()
        

    @unittest.skipIf(True, '')
    def test_contact_search(self):
        self.plot_solution('mesh')

        neighborList = get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, 5)

        def compute_overlap_area(segB, neighborSegsA):
            def compute_area_for_segment_pair(segB, indexA):
                segA = self.segmentConnsA[indexA]
                coordsSegB = self.mesh.coords[segB] + self.disp[segB]
                coordsSegA = self.mesh.coords[segA] + self.disp[segA]
                return MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: 1.0, 1e-9)
            
            return np.sum(vmap(compute_area_for_segment_pair, (None,0))(segB, neighborSegsA))
        
        totalSum = np.sum(vmap(compute_overlap_area)(self.segmentConnsB, neighborList))
    
        self.assertNear(totalSum, 0.3, 5)


    def test_contact_constraints(self):
        self.plot_solution('mesh')

        neighborList = get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, 5)

        def compute_nodal_gap_area(segB, neighborSegsA):
            nodeLeft = segB[0]
            nodeRight = segB[1]
            def compute_quantities_for_segment_pair(segB, indexA):
                segA = self.segmentConnsA[indexA]
                coordsSegB = self.mesh.coords[segB] + self.disp[segB]
                coordsSegA = self.mesh.coords[segA] + self.disp[segA]

                gapAreaLeft = MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: gap * (1.0-xiA), 1e-9)
                areaLeft = MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: (1.0-xiA), 1e-9)
                gapAreaRight = MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: gap * xiA, 1e-9)
                areaRight = MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: xiA, 1e-9)
                return gapAreaLeft, areaLeft, gapAreaRight, areaRight

            gapAreaLeft, areaLeft, gapAreaRight, areaRight = vmap(compute_quantities_for_segment_pair, (None,0))(segB, neighborSegsA)
            return nodeLeft, np.sum(gapAreaLeft), np.sum(areaLeft), nodeRight, np.sum(gapAreaRight), np.sum(areaRight)

        nodesLeft, gapsLeft, areasLeft, nodesRight, gapsRight, areasRight = vmap(compute_nodal_gap_area)(self.segmentConnsB, neighborList)
    
        nodalGapField = np.zeros(self.disp.shape[0])
        nodalAreaField = np.zeros(self.disp.shape[0])

        nodalGapField = nodalGapField.at[nodesLeft].add(gapsLeft)
        nodalGapField = nodalGapField.at[nodesRight].add(gapsRight)
        nodalAreaField = nodalAreaField.at[nodesLeft].add(areasLeft)
        nodalAreaField = nodalAreaField.at[nodesRight].add(areasRight)

        nodesB = np.unique(np.concatenate(self.segmentConnsB))

        print('nodes b = ', nodesB)

        nodalGapField = vmap(lambda x, d : np.where( d > 0, x / d, 0.0))(nodalGapField, nodalAreaField)

        print('nodal area field = ', nodalAreaField[nodesB])
        print('nodal gap field = ', nodalGapField[nodesB])
        print('coords = ', self.mesh.coords[nodesB])

        print('sum area = ', np.sum(nodalAreaField), np.sum(nodalAreaField[nodesB]), np.sum(nodalAreaField[~nodesB]))



if __name__ == '__main__':
    unittest.main()

