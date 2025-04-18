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
#import numpy as onp
#from scipy.sparse import linalg
#from typing import Callable, NamedTuple

# # initiate test array size
# numElemRow = 1
# numPropsColm = 6

# # initiate test array of elements and parameters
# #props = np.full((numElemRow,numPropsColm),8)
# props = np.array([[3,2],[1,4]])

# # print for sanity the number of elements and the props array
# print(np.shape(props))
# print(props)

# #create a function to output whether the vmap is 0 or None
# def vmap_Prop_Test(propArray):
#     numElem = np.shape(propArray)[0]
#     numProp = np.shape(propArray)[1]
#     if numElem > 1:
#         vmapPropValue = 0
#     else:
#         vmapPropValue = 'None'
#     return vmapPropValue

# # print output of vmap_Prop_Test for sanity check
# print(vmap_Prop_Test(props))

# print(np.array([[3,2],[1,4]]))


# props = {
#         'elastic modulus': [1.0, 1.5, 2],
#         'poisson ratio'  : [0.48, 0.48, 0.48]
#         }

# print(props[0])


# def param_function(x, y, z, mult):
#     param = [0]*len(x)
#     for i in range(len(x)):
#         x_coord = x[i]
#         y_coord = y[i]
#         z_coord = z[i]
#         #param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)/10 
#         param[i] =  mult * 1/(5*np.sqrt(x_coord**2 + y_coord**2 + z_coord**2))
#     return param

# # create test props array
# a = np.array([[1, 0.48], [1.5, 0.48], [2, 0.48], [2.5, 0.48]])
# print(np.shape(a))
# print(a)

# # transpose 
# b = np.transpose(a)
# print(b)

# # repeat all props 3 times, along the column 1, 2 -> 1, 1, 1, 2, 2, 2
# c = np.repeat(b,3,axis=1)
# print(c)

# # reshape to create 3D array with each row is an element, each layer is a prop
# d = c.reshape((np.shape(a)[1],np.shape(a)[0],3))
# print(d)




def drumhead_height(n, k, distance, angle, t):
   kth_zero = special.jn_zeros(n, k)[-1]
   return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)
theta = np.r_[0:2*np.pi:50j]
radius = np.r_[0:1:50j]
x = np.array([r * np.cos(theta) for r in radius])
y = np.array([r * np.sin(theta) for r in radius])
z = np.array([drumhead_height(2, 1, r, theta, 0) for r in radius])



# fig, ax = plt.subplots()
# CS = ax.contourf(x, y, z)
# fig.colorbar(CS)
# fig = plt.figure()
# ax = fig.add_axes(rect=(0, 0.05, 0.95, 0.95), projection='3d')
# ax = fig.add_axes(rect=(0, 0.05, 0.95, 0.95),projection = '3d')
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_xticks(np.arange(-1, 1.1, 0.5))
# ax.set_yticks(np.arange(-1, 1.1, 0.5))
# ax.set_zlabel('Z')


# plt.savefig('bessel_fuction_test.png')
# plt.close()



#########################################

def drumhead_height(n, k, distance, angle, t):
   kth_zero = special.jn_zeros(n, k)[-1]
   return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)

x = np.linspace(-1,1,200)
y = x[:]
xv, yv = np.meshgrid(x,y)

z = np.array([drumhead_height(0, 3, 1*np.sqrt((X)**2 + (y-1)**2), np.atan2((y-1),X), 0) for X in x])


# check if the root is at a = 1, i.e. r = 1
# for i in range(len(x)):
#    for j in range(len(y)):
#        if x[i]**2 + y[j]**2 < 1.001 and x[i]**2 + y[j]**2 > 0.999:
#             boundary = drumhead_height(0, 3, 1*np.sqrt(x[i]**2 + y[j]**2), np.atan2(y[j],x[i]), 0)
#             print(boundary)


# print(np.amin(z))
# print(np.amax(z))
# print(np.size(z))
# print(np.shape(z))

fig, ax = plt.subplots()
CS = ax.contourf(x, y, z, 50)
fig.colorbar(CS)

# theta = np.r_[0:2*np.pi:50j]
# r = 1 

# x = r*np.cos(theta)
# y = r*np.sin(theta)

# ax.plot(x,y,linewidth=1,color = 'black',linestyle='--')

# fig = plt.figure()
# ax = fig.add_axes(rect=(0, 0.05, 0.95, 0.95), projection='3d')
# ax = fig.add_axes(rect=(0, 0.05, 0.95, 0.95),projection = '3d')
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_xticks(np.arange(-1, 1.1, 0.5))
# ax.set_yticks(np.arange(-1, 1.1, 0.5))
# ax.set_zlabel('Z')


# plt.savefig('bessel_fuction_test_3.png')
plt.close()











############################################################
meshFile = './ellipse_test_LD_bessel_2.exo'
outputFile = './output_ellipse_LD_bessel_2.exo'
inputMesh = ReadExodusMesh.read_exodus_mesh(meshFile)
# outputMesh = ReadExodusMesh.read_exodus_mesh(outputFile)
new_exo = exodus.exodus(meshFile, mode='a', array_type='numpy')
props = ReadExodusMesh.read_exodus_mesh_element_properties(meshFile, ['light_dose', 'nu'], blockNum=1)
# print(inputMesh.coords)
# print(len(props))

x = inputMesh.coords[0][0] 
# print(x)
# print(len(inputMesh.coords))
for i in range(len(inputMesh.coords)):
#    y[i] = inputMesh.coords[i][0]
    y = y.at[i].set(inputMesh.coords[i][0])
# y = [lambda arg = x: inputMesh.coords[arg][0] for x in range(len(inputMesh.coords))]
# print(y)
# print(len(y))














# create function that converts light dose from dosage to grayscale, linear within the range given
def midGray(lightDose):
    minLD = 20
    maxLD = 1000
    minGray = 50
    maxGray = 255
    gray = ((maxGray - minGray)/(maxLD-minLD))*(lightDose - minLD) + minLD
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


# print(minX.item(0))
# print(maxX.item(0))
# print(minY)
# print(maxY)


# print center for testing
# print(printCenter)

# Testing on how to assign grayscale to pixels
x_coords, y_coords, z_coords = new_exo.get_coords()
elem_conn, num_blk_elems, num_elem_nodes = new_exo.get_elem_connectivity(1)

    # initiate element centroid array
element_centroid = [0] * new_exo.num_elems()

# area calcs function
def inTriangle(xcoordinates, ycoordinates, p):
    aT = abs((xcoordinates[0]*(ycoordinates[1] - ycoordinates[2])
          + xcoordinates[1]*(ycoordinates[2] - ycoordinates[0])
          + xcoordinates[2]*(ycoordinates[0] - ycoordinates[1]))/2)

    a1 = abs((p[0]*(ycoordinates[1] - ycoordinates[2])
          + xcoordinates[1]*(ycoordinates[2] - p[1])
          + xcoordinates[2]*(p[1] - ycoordinates[1]))/2)
    
    a2 = abs((xcoordinates[0]*(p[1] - ycoordinates[2])
          + p[0]*(ycoordinates[2] - ycoordinates[0])
          + xcoordinates[2]*(ycoordinates[0] - p[1]))/2)
    
    a3 = abs((xcoordinates[0]*(ycoordinates[1] - p[1])
          + xcoordinates[1]*(p[1] - ycoordinates[0])
          + p[0]*(ycoordinates[0] - ycoordinates[1]))/2)

    # print([aT, round(a1+a2+a3)])
    # print(round(aT))
    # note: added the -3 because the areas should be equal, but were coming in with very small differences,
    # causing whole elements to be left out.
    # if p[0] in xcoordinates//1:
    #     node = np.where(xcoordinates//1 == p[0])
    #     if (ycoordinates//1)[node] == p[1]:
    #         return 1
    if (a1 + a2 + a3)//1 > aT:
        return 0
    else:
        return 1


def inTriangle2(xcoordinates, ycoordinates, p):
    aT = abs((xcoordinates[0]*(ycoordinates[1] - ycoordinates[2])
          + xcoordinates[1]*(ycoordinates[2] - ycoordinates[0])
          + xcoordinates[2]*(ycoordinates[0] - ycoordinates[1]))/2)

    a1 = abs((p[:,0]*(ycoordinates[1] - ycoordinates[2])
          + xcoordinates[1]*(ycoordinates[2] - p[:,1])
          + xcoordinates[2]*(p[:,1] - ycoordinates[1]))/2)
    
    a2 = abs((xcoordinates[0]*(p[:,1] - ycoordinates[2])
          + p[:,0]*(ycoordinates[2] - ycoordinates[0])
          + xcoordinates[2]*(ycoordinates[0] - p[:,1]))/2)
    
    a3 = abs((xcoordinates[0]*(ycoordinates[1] - p[:,1])
          + xcoordinates[1]*(p[:,1] - ycoordinates[0])
          + p[:,0]*(ycoordinates[0] - ycoordinates[1]))/2)
    
    aTotal = a1 + a2 + a3
    # print(aTTest)
    # print(aTotal)
    # print(len(aTotal))
    return p[np.where(aTotal-10<=aT//1),:]



# find the centroid of each element
# print(new_exo.num_elems())
for e in range(2): #range(new_exo.num_elems()):
    print(e)
    # create an array of the of the nodes that are in element e
    nodes_in_element = elem_conn[num_elem_nodes*(e):num_elem_nodes*e + num_elem_nodes]
    # assign the x, y, and z coordinates of each node in element e
    node_x_coords = x_coords[nodes_in_element-1]
    node_y_coords = y_coords[nodes_in_element-1]
    node_z_coords = z_coords[nodes_in_element-1]

    grayValue = int(round(midGray(props[e][0])))
    # print(grayValue)

    minXLocal = min(node_x_coords)*pixmm + printCenter[0]
    minYLocal = min(node_y_coords)*pixmm + printCenter[1]
    maxXLocal = max(node_x_coords)*pixmm + printCenter[0]
    maxYLocal = max(node_y_coords)*pixmm + printCenter[1]

    # print(minXLocal)
    # print(maxXLocal)

    # print(pixmm*node_x_coords + printCenter[0])
    # print(pixmm*node_y_coords + printCenter[1])
    xScaled = pixmm*node_x_coords + printCenter[0]
    yScaled = pixmm*node_y_coords + printCenter[1]

    # x = np.array([0.26, 0.58 ,0.61 ])
    # y = np.array([0.24385, 0.18651 , 0.7231])
    z = np.ones(3)*grayValue
    # print(z)
    X = np.linspace(int(minXLocal), int(maxXLocal), int(max(xScaled))-int(min(xScaled))+1)
    Y = np.linspace(int(minYLocal), int(maxYLocal), int(max(yScaled))-int(min(yScaled))+1)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    test = list(zip(xScaled,yScaled))
    # print(test)
    interpRes = interp.LinearNDInterpolator(test, z)
    # print(interpRes(X,Y))
    Z = interpRes(X,Y)
    # print(X)
    # print(Y)
    # print(Z)
    # Z[np.isnan(Z)] = 0
    nanValues = np.where(np.isnan(Z))
    # print(nanValues[0])
    # print(nanValues[1])
    ZTest = Z[:]
    # print(ZTest)
    ZTest[nanValues[0],nanValues[1]] = [0]
    # print(ZTest)
    plt.pcolormesh(X,Y,Z)
    # plt.colorbar()
    plt.axis("equal")
    buildImage = buildImage.at[np.ravel(X),np.ravel(Y)].set(grayValue)
    #################################################################################
    # xrange = np.linspace(int(minXLocal),int(maxXLocal),int(maxXLocal)-int(minXLocal)+1)
    # yrange = np.linspace(int(minYLocal),int(maxYLocal),int(maxYLocal)-int(minYLocal)+1)

    # # create array of all points in the search region.
    # xv, yv = np.meshgrid(xrange,yrange)
    # xflat = np.ravel(xv)
    # yflat = np.ravel(yv)
    # combXY = np.append(xflat,yflat)
    # points = np.reshape(combXY,(-1,2),order='F')
    # print(points)

    # print(points[:,0])
    
    # a1Test = abs((points[:,0]*(yScaled[1] - yScaled[2])
    #       + xScaled[1]*(yScaled[2] - points[:,1])
    #       + xScaled[2]*(points[:,1] - yScaled[1]))/2)

    # a2Test = abs((xScaled[0]*(points[:,1] - yScaled[2])
    #       + points[:,0]*(yScaled[2] - yScaled[0])
    #       + xScaled[2]*(yScaled[0] - points[:,1]))/2)
    
    # a3Test = abs((xScaled[0]*(yScaled[1] - points[:,1])
    #       + xScaled[1]*(points[:,1] - yScaled[0])
    #       + points[:,0]*(yScaled[0] - yScaled[1]))/2)
    
    # aTTest = abs((xScaled[0]*(yScaled[1] - yScaled[2])
    #       + xScaled[1]*(yScaled[2] - yScaled[0])
    #       + xScaled[2]*(yScaled[0] - yScaled[1]))/2)
    
    # aTotal = a1Test + a2Test + a3Test
    # print(aTTest)
    # print(aTotal)
    # print(len(aTotal))
    # validPoints = inTriangle2(xScaled,yScaled,points)
    # # print(validPoints[0][:,0])

    # xValid = validPoints[0][:,0]
    # yValid = validPoints[0][:,1]

    # # testing changing all pixels at once in the element
    # for i in range(len(xValid)):
    #     # print([int(xValid[i]), int(yValid[i])])
    #     buildImage = buildImage.at[int(yValid[i]),int(xValid[i])].set(grayValue)

    #################################################################################



    # # calculate if a pixel is in the element
    # for i in range(int(minXLocal-1),int(maxXLocal+1)):
    #     for j in range(int(minYLocal-1),int(maxYLocal+1)):
    #         # print([i,j])
    #         if inTriangle(xScaled, yScaled, [i,j]) == 1:
    #             buildImage = buildImage.at[j,i].set(grayValue)
    #             # print([i,j,grayValue])

            





    # find the centroid of element i, assign the coordinates to element_centroid
    # element_centroid[e] = [np.average(node_x_coords), np.average(node_y_coords), np.average(node_z_coords)]


#print(buildImage[int(minY):int(maxY),int(minX):int(maxX)])
plt.xlim([0, 1920])
plt.ylim([0, 1080])
plt.savefig('elementInterp.png',dpi=600)

plt.close()

xFig = np.arange(0, xheight, 1)
yFig = np.arange(0, yheight, 1)

fig, ax = plt.subplots()
meshImage = ax.pcolormesh(xFig, yFig, buildImage)
fig.colorbar(meshImage)
ax.set_aspect('equal')
# scatter to find element nodes for troubleshooting
# area = [0.01, 0.01, 0.01]
# plt.scatter(node_x_coords*pixmm + printCenter[0],node_y_coords*pixmm + printCenter[1], s=area)

plt.savefig('mesh_to_Image_Test_5.png', dpi = 600)
plt.close()











new_exo.close()






