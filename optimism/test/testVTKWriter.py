import os
import unittest
import warnings

import jax.numpy as np

from optimism.VTKWriter import VTKWriter, VTKFieldType, VTKDataType
from optimism import Mesh
from optimism.test import MeshFixture


class TestVTKWriter(MeshFixture.MeshFixture):
    baseFileName = 'output'

    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        self.mesh, self.dofValues = self.create_mesh_and_disp(self.Nx, self.Ny,
                                                              xRange, yRange,
                                                              lambda x : self.targetDispGrad.dot(x))

        self.writer = VTKWriter(self.mesh, baseFileName=self.baseFileName)
        scalarField = self.mesh.coords[:,0]
        self.writer.add_nodal_field(name='temperature', nodalData=scalarField, fieldType=VTKFieldType.SCALARS)
        vectorField = self.mesh.coords
        self.writer.add_nodal_field(name='displacement', nodalData=vectorField, fieldType=VTKFieldType.VECTORS)
        nnodes = self.mesh.coords.shape[0]
        tensorField = np.arange(nnodes*3*3).reshape(nnodes,3,3)
        self.writer.add_nodal_field(name='stress', nodalData=tensorField, fieldType=VTKFieldType.TENSORS)
        nelements = self.mesh.conns.shape[0]
        cellID = np.arange(nelements).reshape((nelements,))
        self.writer.add_cell_field(name='cell_id', cellData=cellID, fieldType=VTKFieldType.SCALARS, dataType=VTKDataType.INT)

        
    def tearDown(self):
        if os.path.exists(self.baseFileName + '.vtk'):
            os.remove(self.baseFileName + '.vtk')


    def test_vtk_write(self):
        self.writer.write()
        self.assertTrue(os.path.exists(self.baseFileName + '.vtk'))


    def test_vtk_no_warning_for_inconsistent_sizes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.writer.write()



class TestVTKWriterHigherOrder(MeshFixture.MeshFixture):
    baseFileName = 'output'

    def setUp(self):
        self.Nx = 3
        self.Ny = 3
        xRange = [0.,1.]
        yRange = [0.,1.]
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        self.simplexMesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny,
                                                        xRange, yRange,
                                                        lambda x : self.targetDispGrad.dot(x))

        
    def tearDown(self):
        if os.path.exists(self.baseFileName + '.vtk'):
            os.remove(self.baseFileName + '.vtk')

            
    def make_high_order_writer_with_fields(self, mesh):     
        self.writer = VTKWriter(mesh, baseFileName=self.baseFileName)
        
        scalarField = mesh.coords[:,0]
        self.writer.add_nodal_field(name='temperature', nodalData=scalarField, fieldType=VTKFieldType.SCALARS)
        vectorField = mesh.coords
        self.writer.add_nodal_field(name='displacement', nodalData=vectorField, fieldType=VTKFieldType.VECTORS)
        nnodes = mesh.coords.shape[0]
        tensorField = np.arange(nnodes*3*3).reshape(nnodes,3,3)
        self.writer.add_nodal_field(name='stress', nodalData=tensorField, fieldType=VTKFieldType.TENSORS)
        nelements = mesh.conns.shape[0]
        cellID = np.arange(nelements).reshape((nelements,))
        self.writer.add_cell_field(name='cell_id', cellData=cellID, fieldType=VTKFieldType.SCALARS, dataType=VTKDataType.INT)

        
    def test_vtk_writer_on_quadratic_elements(self):
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.simplexMesh, order=2)
        self.make_high_order_writer_with_fields(mesh)
        self.writer.write()
        self.assertTrue(os.path.exists(self.baseFileName + '.vtk'))

        
    def test_vtk_writer_no_inconsistent_sizes_with_quadratic_elements(self):
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.simplexMesh, order=2)
        self.make_high_order_writer_with_fields(mesh)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.writer.write()


    def test_vtk_writer_on_cubic_elements(self):
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.simplexMesh, order=3)
        self.make_high_order_writer_with_fields(mesh)
        self.writer.write()
        self.assertTrue(os.path.exists(self.baseFileName + '.vtk'))

        
    def test_vtk_writer_no_inconsistent_sizes_with_cubic_elements(self):
        mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.simplexMesh, order=3)
        self.make_high_order_writer_with_fields(mesh)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.writer.write()

    

if __name__ == '__main__':
    unittest.main()
