#from collections import namedtuple
#from jax import grad
from jax import jit
from optimism import EquationSolver
#from plato_optimism import exodus_writer as ExodusWriter
#from plato_optimism import GradUtilities
#from optimism import FunctionSpace
#from optimism import Interpolants
#from optimism import Mechanics
#from optimism import Mesh
#from optimism import Objective
#from optimism import QuadratureRule
from optimism import ReadExodusMesh
#from optimism import SparseMatrixAssembler
#from optimism.FunctionSpace import DofManager
#from optimism.FunctionSpace import EssentialBC
#from optimism.inverse import MechanicsInverse
#from optimism.inverse import AdjointFunctionSpace
#from optimism.material import Neohookean_VariableProps
import jax.numpy as np
import scipy as sp
import scipy.special as special
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import exodus3 as exodus
import skimage as skim 
import timeit
start = timeit.default_timer()
############################################################
meshFile = './ellipse_test_LD_bessel_2.exo'
outputFile = './output_ellipse_LD_bessel_2.exo'
inputMesh = ReadExodusMesh.read_exodus_mesh(meshFile)
# outputMesh = ReadExodusMesh.read_exodus_mesh(outputFile)
new_exo = exodus.exodus(meshFile, mode='a', array_type='numpy')
props = ReadExodusMesh.read_exodus_mesh_element_properties(meshFile, ['light_dose', 'nu'], blockNum=1)

# create function that converts light dose from dosage to grayscale, linear within the range given
def midGray(lightDose):
    minLD = 20
    maxLD = 1000
    minGray = 50
    maxGray = 255
    gray = ((maxGray - minGray)/(maxLD-minLD))*(lightDose - minLD) + minGray
    return gray # (255 / (maxLD-minLD))*(lightDose - minLD)

# define build plate size in pixels and mm
yheight = 1080 # pixels
xheight = 1920 # pixels
yheightmm = 65 # mm
xheightmm = 115 # mm
ypixmm = yheight/yheightmm
xpixmm = xheight/xheightmm

# constant to convert mm dimensions to pixels
pixmm = 30*(ypixmm + xpixmm)/2

# initiate the build plate pixel space
buildPlate = np.zeros([yheight, xheight]) # [pixels, pixels]
buildImage = buildPlate[:]

# define where the print will be centered
printCenter = [xheight/2, yheight/2] # [pixels, pixels]

# Define range to search for elements
minX = np.round(-1.01*pixmm + printCenter[0])
maxX = np.round(1.01*pixmm + printCenter[0])
minY = np.round(-1.01*pixmm + printCenter[1])
maxY = np.round(1.01*pixmm + printCenter[1])

# Testing on how to assign grayscale to pixels
x_coords, y_coords, z_coords = new_exo.get_coords()
elem_conn, num_blk_elems, num_elem_nodes = new_exo.get_elem_connectivity(1)

    # initiate element centroid array
element_centroid = [0] * new_exo.num_elems()

img = np.zeros((yheight, xheight), dtype = np.uint8)
rr, cc = skim.draw.rectangle((0,0), extent=(yheight, xheight))
img = img.at[rr,cc].set(0)

# find the centroid of each element

for e in range(new_exo.num_elems()):
    # print(e)
    # create an array of the of the nodes that are in element e
    nodes_in_element = elem_conn[num_elem_nodes*(e):num_elem_nodes*e + num_elem_nodes]
    # assign the x, y, and z coordinates of each node in element e
    node_x_coords = x_coords[nodes_in_element-1]
    node_y_coords = y_coords[nodes_in_element-1]
    node_z_coords = z_coords[nodes_in_element-1]

    # find the centroid of element i, assign the coordinates to element_centroid
    element_centroid[e] = [np.average(node_x_coords), np.average(node_y_coords), np.average(node_z_coords)]

    xScaled = pixmm*node_x_coords + printCenter[0]
    yScaled = pixmm*node_y_coords + printCenter[1]
    
    grayValue = int(round(midGray(props[e][0])))
    # print(grayValue)

    rr, cc = skim.draw.polygon(yScaled,xScaled)
    img = img.at[rr,cc].set(grayValue)


img = img[::-1,:]
skim.io.imsave('image_process_test1.png',img)
print(timeit.default_timer()-start)
new_exo.close()