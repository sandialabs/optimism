from optimism import ReadExodusMesh
import jax.numpy as jnp
import numpy as np
import skimage as skim 
import timeit
import os
import glob
import meshio

# start a timer for run-time
start = timeit.default_timer()

###################################################################################################

# define a function that collects all the directories 
# in the defined directory
def get_workdirs(directory,pattern):
    return glob.glob(os.path.join(directory,pattern))
    
# collect all workdirs in the current workdir
workdirs = get_workdirs('./','workdir*')

# pull the first output-002.vtk file from the second-to
# last workdir. Second to last, since the last workdir 
# may have not converged. If plato converges, change -2
# to -1
meshFile = workdirs[-2] + '/output-002.vtk'

# read the mesh vtk file
mesh = meshio.read(meshFile)

# read the original mesh for the nodal and element data
new_exo = ReadExodusMesh.read_exodus_mesh('./EXO_files/ellipse_test_Seeded.exo')

# define the properties (density values) from the vtk file
props = mesh.cell_data['element_property_field'][0]

# create function that converts plato-density to grayscale, linear within the range given
# This should be changed to match the appropriate behavior and conversion
# ENSURE THIS MATCHES WITH THE CONVERSION IN THE MATERIAL MODEL
def midGray(property):
    minLD = 0
    maxLD = 1
    minGray = 50
    maxGray = 255
    gray = ((maxGray - minGray)/(maxLD-minLD))*(property - minLD) + minGray
    return gray

# define build plate size in pixels and mm
# The mm are necessary to ensure the mesh is appropriately sized 
yheight = 1080 # pixels
xheight = 1920 # pixels
yheightmm = 65 # mm
xheightmm = 115 # mm
ypixmm = yheight/yheightmm
xpixmm = xheight/xheightmm

# constant to convert mm dimensions to pixels
# the times 10 increases the size of the input mesh, a magnification transformation
pixmm = 10*(ypixmm + xpixmm)

# initiate the build plate pixel space
buildPlate = jnp.zeros([yheight, xheight]) # [pixels, pixels]
buildImage = buildPlate[:]

# define where the print will be centered
# note that this is based on the origin of the mesh file.
# typically the mesh is centered about a (0,0) origin.
printCenter = [xheight/2, yheight/2] # [pixels, pixels]

# Define range to search for elements
# this ensures that the code doesn't search the whole build-plate
# for valid pixels, just where the mesh is.
minX = jnp.round(-1.01*pixmm + printCenter[0])
maxX = jnp.round(1.01*pixmm + printCenter[0])
minY = jnp.round(-1.01*pixmm + printCenter[1])
maxY = jnp.round(1.01*pixmm + printCenter[1])

# Testing on how to assign grayscale to pixels
x_coords = new_exo.coords[:,0]
y_coords = new_exo.coords[:,1]

# define the nodes that compose each element
elem_conn = new_exo.conns

# define the number of nodes per element
num_elem_nodes = len(elem_conn[0])

# define the number of elements in the mesh
num_elem = np.shape(elem_conn)

# initiate element centroid array
# element_centroid = np.zeros((num_elem[0],2))

# inintialize the image array, everything set to 0
img = jnp.zeros((yheight, xheight), dtype = jnp.uint8)
rr, cc = skim.draw.rectangle((0,0), extent=(yheight, xheight))
img = img.at[rr,cc].set(0)


# define the nodes that make up an element, e, scale them to match the 
# build plate size, and draw a polygon (triangle, most commonly) with 
# the corresponding grayscale value
for e in range(np.shape(elem_conn)[0]):
    # print(e)

    # create an array of the of the nodes that are in the current element e
    # nodes_in_element = elem_conn[num_elem_nodes*(e):num_elem_nodes*e + num_elem_nodes]
    nodes_in_element = elem_conn[e]

    # define the x, y, and z coordinates of each node in the current element e
    node_x_coords = x_coords[nodes_in_element-0]
    node_y_coords = y_coords[nodes_in_element-0]
    # node_z_coords = z_coords[nodes_in_element-1]

    # find the centroid of the current element e, assign the coordinates to element_centroid
    # element_centroid[e] = [np.average(node_x_coords), np.average(node_y_coords), np.average(node_z_coords)]
    # element_centroid[e] = [jnp.average(node_x_coords), jnp.average(node_y_coords)]

    # scale the values of the element vertices from physical dimensions to pixel corrdinates
    xScaled = pixmm*node_x_coords + printCenter[0]
    yScaled = pixmm*node_y_coords + printCenter[1]
    
    # calculate the grayscale value using the midGray function using the element light dosage
    # in future iterations, calculate this outside of this loop and pull the value for each element
    grayValue = int(round(midGray(props[e][0])))
    # print(grayValue)

    # draw a triangle whose vertices are the nodes that define the current element
    rr, cc = skim.draw.polygon(np.array(yScaled),np.array(xScaled))

    # assign all the pixels within the triangle to the current element grayscale value
    img = img.at[rr,cc].set(grayValue)

# flip the rows, since image processing has y-positive pointing down, the origin in the upper left corner.
# this flips the image in the y-direction, making the origin in the lower left, y-positive pointing up
img = img[::-1,:]

############################### change the name of the saved image #########################################
skim.io.imsave('./Images/image_process_test1.png',img)

# print the time it took to process the image, for testing!!!
print('Image complete: ' + str(round(timeit.default_timer()-start,2)) + ' seconds')

# close the exo file
# new_exo.close()