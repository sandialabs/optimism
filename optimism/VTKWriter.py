import numpy as np
from enum import Enum, auto
from collections import namedtuple
import warnings


def write_matrix_as_table(A):
    # it's insane that numpy requires this kind
    # of hacking to write out a matrix as a table
    return '\n'.join(' '.join('{}'.format(x) for x in y) for y in A)



class VTKFieldType(Enum):
    SCALARS = auto()
    VECTORS = auto()
    TENSORS = auto()
    

class VTKDataType(Enum):
    BIT = 'bit'
    UNSIGNED_CHAR = 'unsigned_char'
    CHAR = 'char'
    UNSIGNED_SHORT = 'unsigned_short'
    SHORT = 'short'
    UNSIGNED_INT = 'unsigned_int'
    INT = 'int'
    UNSIGNED_LONG = 'unsigned_long'
    LONG = 'long'
    FLOAT = 'float'
    DOUBLE = 'double'


def default_values(fieldType, dataType):
    if dataType==VTKDataType.FLOAT or  dataType==VTKDataType.DOUBLE:
        if fieldType==VTKFieldType.SCALARS:
            return 0.
        elif fieldType==VTKFieldType.VECTORS:
            return np.array([0.0, 0.0, 0.0])
        elif fieldType==VTKFieldType.TENSORS:
            return np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    else:
        if fieldType==VTKFieldType.SCALARS:
            return 0
        elif fieldType==VTKFieldType.VECTORS:
            return np.array([0, 0, 0])
        elif fieldType==VTKFieldType.TENSORS:
            return np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    
# Try to do 2D/3D -> 3D in one code path
#def make_data_3Db(A):
#    ndims = A.ndim
#    A3DShape = [A.shape[0]] + [3 for i in range(ndims)]
#    A3D = np.zeros(A3DShape, A.dtype)
#    # on LHS need to get 0:2 for every dimension after first
#    # maybe use slice()?



class VTKWriter:
    def __init__(self, mesh, baseFileName='output'):
        self.mesh = mesh
        self.fileName = baseFileName + '.vtk'
        self.nodalFields = {}
        self.cellFields = {}
        self.vtkFormat = 'ASCII'

        self.spheres = []
        self.sphereRadii = []

        # Legacy vtk can only plot linear and quadratic elements.
        # We will plot quadratic elements as is, and
        # all other elements will be converted to linear
        if mesh.parentElement.degree==2:
            self.outputNodes = np.arange(mesh.coords.shape[0])
            self.elConn = np.array([0, 2, 5, 1, 4, 3], dtype=np.int_)
            self.vtkCellType = 22
        else:
            self.outputNodes = mesh.simplexNodesOrdinals
            self.elConn = mesh.parentElement.vertexNodes
            self.vtkCellType = 5
        
        
    def add_sphere(self, x, radius):
        self.spheres.append( np.array([x[0], x[1], 0.0]) )
        self.sphereRadii.append( radius )


    def add_nodal_field(self, name, nodalData, fieldType, dataType=VTKDataType.DOUBLE):
        nnodes = self.mesh.coords[self.outputNodes].shape[0]
        nodalData = nodalData[self.outputNodes]
        if nodalData.shape[0] == nnodes:
            nodalData3D = self._check_and_format_data(np.array(nodalData), fieldType)
            self.nodalFields[name] = self.VTKFieldRecord(data=nodalData3D, fieldType=fieldType, dataType=dataType)
        else:
            warnings.warn('VTKWriter: Added field has incorrect shape or number of '
                          'entries\n'
                          'Skipping output of nodal field {0}'.format(name))
            

    def add_cell_field(self, name, cellData, fieldType, dataType=VTKDataType.DOUBLE):
        nelements = self.mesh.conns.shape[0]
        if cellData.shape[0] == nelements:
            cellData3D = self._check_and_format_data(np.array(cellData), fieldType)
            self.cellFields[name] = self.VTKFieldRecord(data=cellData3D, fieldType=fieldType,  dataType=dataType)
        else:
            warnings.warn('VTKWriter: Added field has incorrect shape or number of '
                          'entries\n'
                          'Skipping output of cell field {0}'.format(name))

        
    def write(self):
        try:
            vtkFile = open(self.fileName, 'w')
        except OSError as err:
            warnings.warn('VTKWriter: Unable to open file: {0}'.format(err))
            return

        self._write_header(vtkFile)
        self._write_coordinate_data(vtkFile)
        self._write_cell_connectivity(vtkFile)
        self._write_cell_types(vtkFile)
        self._write_nodal_fields(vtkFile)
        self._write_cell_fields(vtkFile)
        vtkFile.close()

    
    # private
    
    VTKFieldRecord = namedtuple('VTKFieldRecord', ['data', 'fieldType', 'dataType'])

    
    def _check_and_format_data(self, field, fieldType):
        if fieldType==VTKFieldType.SCALARS:
            assert( (field.ndim == 1) or \
                    (field.ndim == 2 and field.shape[1] == 1) )
            field3D = field.reshape((-1,1))
        elif fieldType==VTKFieldType.VECTORS:
            assert(field.ndim == 2)
            field3D = np.zeros((field.shape[0], 3), field.dtype)
            spatialDim = field.shape[1]
            field3D[:,0:spatialDim] = field
        elif fieldType==VTKFieldType.TENSORS:
            assert(field.ndim == 3 and \
                   field.shape[1] == field.shape[2])
            spatialDim = field.shape[1]
            field3D = np.zeros((field.shape[0], 3, 3), field.dtype)
            field3D[:,0:spatialDim,0:spatialDim] = field
            field3D = field3D.reshape((-1,3))
        return field3D
    
    
    def _write_header(self, vtkFile):
        vtkFile.write('# vtk DataFile Version 3.0\n')
        vtkFile.write('Written from jax-fem\n')
        vtkFile.write(self.vtkFormat + '\n')
        vtkFile.write('DATASET UNSTRUCTURED_GRID\n')
        
        
    def _write_coordinate_data(self, vtkFile):
        coords = self.mesh.coords[self.outputNodes]
        nnodes = coords.shape[0]
        vtkFile.write('POINTS {} double\n'.format(nnodes + len(self.spheres)))
        dim = coords.shape[1]
        coords3D = np.zeros((nnodes, 3))
        coords3D[:, 0:dim] = coords
        vtkFile.write(write_matrix_as_table(coords3D))
        vtkFile.write('\n')
        for spherePt in self.spheres:
            ptLine = str(spherePt[0]) + ' ' + str(spherePt[1]) + ' ' + str(spherePt[2]) + '\n'
            vtkFile.write(ptLine)
        
        
    def _write_cell_connectivity(self, vtkFile):
        nelements = self.mesh.conns.shape[0]
        # strong assumption: there is only one element type
        conns = self.mesh.conns[:,self.elConn]
        nNodesPerElement = conns.shape[1]
        nNodesPerElementArray = np.tile(nNodesPerElement,
                                        (nelements,1))
        nvals = conns.size + nNodesPerElementArray.size
        vtkFile.write('CELLS {} {}\n'.format(nelements, nvals))
        vtkConnectivity = np.concatenate((nNodesPerElementArray,conns), axis=1)
        vtkFile.write(write_matrix_as_table(vtkConnectivity))
        vtkFile.write('\n')
        
        
    def _write_cell_types(self, vtkFile):
        nelements = self.mesh.conns.shape[0]
        vtkFile.write('CELL_TYPES {}\n'.format(nelements))
        vtkCellTypes = np.tile(self.vtkCellType, (nelements, 1))
        vtkFile.write(write_matrix_as_table(vtkCellTypes))
        vtkFile.write('\n')

        
    def _write_nodal_fields(self, vtkFile):

        allFieldsAreEmpty = not self.nodalFields
        if not allFieldsAreEmpty or len(self.spheres) > 0:
            coords = self.mesh.coords[self.outputNodes]
            nnodes = coords.shape[0]

            for field in self.nodalFields:
                fieldRecord = self.nodalFields[field]
                for sphere in self.spheres:
                    uNew = np.vstack( (fieldRecord.data,
                                       default_values(fieldRecord.fieldType, fieldRecord.dataType)) )
                    fieldRecord = self.VTKFieldRecord(uNew,
                                                      fieldRecord.fieldType,
                                                      fieldRecord.dataType)
                    



                self.nodalFields[field] = fieldRecord


            if len(self.spheres) > 0:
                nnodes = self.mesh.coords.shape[0]
                vals = np.zeros( (nnodes,) )
                vals = np.hstack( (vals, np.array(self.sphereRadii) ) )
                self.nodalFields['sphere_radius'] = self.VTKFieldRecord(vals.reshape(vals.shape[0],1),
                                                                        VTKFieldType.SCALARS,
                                                                        VTKDataType.DOUBLE)
        
                
            vtkFile.write('POINT_DATA {}\n'.format(nnodes + len(self.spheres)))
            self._write_out_all_fields_in_dict(self.nodalFields, vtkFile)
            
        
    def _write_cell_fields(self, vtkFile):
        allFieldsAreEmpty = not self.cellFields
        if not allFieldsAreEmpty:
            ncells = self.mesh.conns.shape[0]
            vtkFile.write('CELL_DATA {}\n'.format(ncells))
            self._write_out_all_fields_in_dict(self.cellFields, vtkFile)
        
        
    def _write_out_all_fields_in_dict(self, fieldDict, vtkFile):
        for field in fieldDict:
            fieldRecord = fieldDict[field]
            u = fieldRecord.data
            fieldType = fieldRecord.fieldType
            dataType = fieldRecord.dataType
            if fieldType == VTKFieldType.SCALARS:
                vtkFile.write('SCALARS {} {}\n'.format(field, dataType.value))
                vtkFile.write('LOOKUP_TABLE default\n')
            elif fieldType == VTKFieldType.VECTORS:
                vtkFile.write('VECTORS {} {}\n'.format(field, dataType.value));
            elif fieldType == VTKFieldType.TENSORS:
                vtkFile.write('TENSORS {} {}\n'.format(field, dataType.value));
            vtkFile.write(write_matrix_as_table(u))
            vtkFile.write('\n')
