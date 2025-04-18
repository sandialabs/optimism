from jax import jit
from optimism import EquationSolver
from optimism import ReadExodusMesh
import jax.numpy as np
import scipy as sp
import scipy.special as special
import scipy.interpolate as interp
import matplotlib.pyplot as plt
# import exodus3 as exodus
import skimage as skim 
import timeit
import os
import glob
import meshio
start = timeit.default_timer()
############################################################

# collect all the workdir's in the current directory
def get_workdirs(directory,pattern):
    return glob.glob(os.path.join(directory,pattern))
    
# create list of all work directories in current directory
workdirs = get_workdirs('./','workdir*')

# specify the last workdir
meshFile = workdirs[-2] + '/output-0.vtk'
print(meshFile)

# read the 
mesh = meshio.read(meshFile)
print(mesh.cell_data['element_property_field'][0])


# meshConv = "output_conversion.exo"
# mesh.write(meshConv)


# inputMesh = ReadExodusMesh.read_exodus_mesh(meshConv)

# props = ReadExodusMesh.read_exodus_mesh_element_properties(meshFile, ['element_property_field'], blockNum=1)
