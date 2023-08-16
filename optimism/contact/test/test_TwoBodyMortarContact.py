from optimism.JaxConfig import *
from optimism import Mesh
from optimism import QuadratureRule
from optimism import Surface
from optimism.contact import EdgeIntersection
from optimism import FunctionSpace
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
                return MortarContact.integrate_with_mortar(coordsSegB, coordsSegA, MortarContact.compute_average_normal, lambda xiA, xiB, gap: 1.0, 1e-9)
            
            return np.sum(vmap(compute_area_for_segment_pair, (None,0))(segB, neighborSegsA))
        
        totalSum = np.sum(vmap(compute_overlap_area)(self.segmentConnsB, neighborList))
        print('total sum = ', totalSum)
        self.assertNear(totalSum, 1.0, 5)


    def test_contact_constraints(self):
        self.plot_solution('mesh')

        neighborList = MortarContact.get_closest_neighbors(self.segmentConnsA, self.segmentConnsB, self.mesh, self.disp, 5)

        nodalGapField = MortarContact.assemble_area_weighted_gaps(self.mesh.coords, self.disp, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal)
        nodalAreaField = MortarContact.assemble_nodal_areas(self.mesh.coords, self.disp, self.segmentConnsA, self.segmentConnsB, neighborList, MortarContact.compute_average_normal)

        nodalGapField = vmap(lambda x, d : np.where( d > 0, x / d, 0.0))(nodalGapField, nodalAreaField)

        mask = np.ones(len(nodalAreaField), dtype=np.int8)
        nodesB = np.unique(np.concatenate(self.segmentConnsB))
        mask = mask.at[nodesB].set(0)

        self.assertNear(np.sum(nodalAreaField), 1.0, 8)
        self.assertNear(np.sum(nodalAreaField), np.sum(nodalAreaField[nodesB]), 14)
        self.assertNear(np.sum(nodalAreaField[mask]), 0.0, 14)


if __name__ == '__main__':
    unittest.main()

