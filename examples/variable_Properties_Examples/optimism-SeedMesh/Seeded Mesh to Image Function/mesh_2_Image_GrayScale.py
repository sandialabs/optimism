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
meshFile = './EXO_files/ellipse_test_Seeded.exo'
# outputFile = './output_solidTest_3_Seeded_3.exo'
inputMesh = ReadExodusMesh.read_exodus_mesh(meshFile)
# outputMesh = ReadExodusMesh.read_exodus_mesh(outputFile)
new_exo = exodus.exodus(meshFile, mode='a', array_type='numpy')
props = ReadExodusMesh.read_exodus_mesh_element_properties(meshFile, ['density', 'nu'], blockNum=1)

# create function that converts light dose from dosage to grayscale, linear within the range given
# This should be changed to match the appropriate behavior and conversion
def midGray(property):
    minLD = 0
    maxLD = 1
    minGray = 50
    maxGray = 255
    gray = ((maxGray - minGray)/(maxLD-minLD))*(property - minLD) + minGray
    return gray # (255 / (maxLD-minLD))*(lightDose - minLD)

# define build plate size in pixels and mm
# The mm are necessary to ensure the mesh is appropriately sized 
yheight = 1080 # pixels
xheight = 1920 # pixels
yheightmm = 65 # mm
xheightmm = 115 # mm
ypixmm = yheight/yheightmm
xpixmm = xheight/xheightmm

# constant to convert mm dimensions to pixels
# the times 3 increases the size of the input mesh, a magnification
pixmm = 5*(ypixmm + xpixmm)/2

# initiate the build plate pixel space
buildPlate = np.zeros([yheight, xheight]) # [pixels, pixels]
buildImage = buildPlate[:]

# define where the print will be centered
# note that this is based on the origin of the mesh file.
# typically the mesh is centered about a (0,0) origin.
printCenter = [xheight/2, yheight/2] # [pixels, pixels]

# Define range to search for elements
# this ensures that the code doesn't search the whole build-plate
# for valid pixels, just where the mesh is.
minX = np.round(-1.01*pixmm + printCenter[0])
maxX = np.round(1.01*pixmm + printCenter[0])
minY = np.round(-1.01*pixmm + printCenter[1])
maxY = np.round(1.01*pixmm + printCenter[1])

# Testing on how to assign grayscale to pixels
x_coords, y_coords, z_coords = new_exo.get_coords()
elem_conn, num_blk_elems, num_elem_nodes = new_exo.get_elem_connectivity(1)

# initiate element centroid array
element_centroid = [0] * new_exo.num_elems()

# inintialize the image array
img = np.zeros((yheight, xheight), dtype = np.uint8)
rr, cc = skim.draw.rectangle((0,0), extent=(yheight, xheight))
img = img.at[rr,cc].set(0)

# find the centroid of each element

for e in range(new_exo.num_elems()):
    # print(e)

    # create an array of the of the nodes that are in the current element e
    nodes_in_element = elem_conn[num_elem_nodes*(e):num_elem_nodes*e + num_elem_nodes]

    # define the x, y, and z coordinates of each node in the current element e
    node_x_coords = x_coords[nodes_in_element-1]
    node_y_coords = y_coords[nodes_in_element-1]
    node_z_coords = z_coords[nodes_in_element-1]

    # find the centroid of the current element e, assign the coordinates to element_centroid
    element_centroid[e] = [np.average(node_x_coords), np.average(node_y_coords), np.average(node_z_coords)]
    
    # scale the values of the element vertices from physical dimensions to pixel corrdinates
    xScaled = pixmm*node_x_coords + printCenter[0]
    yScaled = pixmm*node_y_coords + printCenter[1]
    
    # calculate the grayscale value using the midGray function using the element light dosage
    # in future iterations, calculate this outside of this loop and pull the value for each element
    grayValue = int(round(midGray(props[e][0])))
    # print(grayValue)

    # draw a triangle whose vertices are the nodes that define the current element
    rr, cc = skim.draw.polygon(yScaled,xScaled)

    # assign all the pixels within the triangle to the current element grayscale value
    img = img.at[rr,cc].set(grayValue)

# flip the rows, since image processing has y-positive pointing down, the origin in the upper left corner.
# this flips the image in the y-direction, making the origin in the lower left, y-positive pointing up
img = img[::-1,:]

############################### change the name of the saved image #########################################
skim.io.imsave('./Images/image_process_test2.png',img)

# print the time it took to process the image, for testing!!!
print(timeit.default_timer()-start)

# close the exo file
new_exo.close()