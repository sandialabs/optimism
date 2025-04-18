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
import matplotlib.pyplot as plt
import exodus3 as exodus
from scipy.interpolate import LinearNDInterpolator
#import numpy as onp
#from scipy.sparse import linalg
#from typing import Callable, NamedTuple













# ############################################################
# meshFile = './ellipse_test_LD_bessel_2.exo'
# outputFile = './output_ellipse_LD_bessel_2.exo'
# inputMesh = ReadExodusMesh.read_exodus_mesh(meshFile)
# # outputMesh = ReadExodusMesh.read_exodus_mesh(outputFile)
# new_exo = exodus.exodus(meshFile, mode='a', array_type='numpy')
# props = ReadExodusMesh.read_exodus_mesh_element_properties(meshFile, ['light_dose', 'nu'], blockNum=1)
# # print(inputMesh.coords)
# # print(len(props))

# # x = inputMesh.coords[0][0] 
# # # print(x)
# # # print(len(inputMesh.coords))
# # for i in range(len(inputMesh.coords)):
# # #    y[i] = inputMesh.coords[i][0]
# #     y = y.at[i].set(inputMesh.coords[i][0])
# # # y = [lambda arg = x: inputMesh.coords[arg][0] for x in range(len(inputMesh.coords))]
# # # print(y)
# # # print(len(y))














# # create function that converts light dose from dosage to grayscale, linear within the range given
# def midGray(lightDose):
#     minLD = 20
#     maxLD = 1000
#     minGray = 50
#     maxGray = 255
#     gray = ((maxGray - minGray)/(maxLD-minLD))*(lightDose - minLD) + minLD
#     return gray # (255 / (maxLD-minLD))*(lightDose - minLD)

# # define build plate size in pixels and mm
# yheight = 1080 # pixels
# xheight = 1920 # pixels
# yheightmm = 65 # mm
# xheightmm = 115 # mm
# ypixmm = yheight/yheightmm
# xpixmm = xheight/xheightmm

# # constant to convert mm dimensions to pixels
# pixmm = 30*(ypixmm + xpixmm)/2

# # initiate the build plate pixel space
# buildPlate = np.zeros([yheight, xheight]) # [pixels, pixels]
# buildImage = buildPlate[:]

# # define where the print will be centered
# printCenter = [xheight/2, yheight/2] # [pixels, pixels]

# # Define range to search for elements
# minX = np.round(-1.01*pixmm + printCenter[0])
# maxX = np.round(1.01*pixmm + printCenter[0])
# minY = np.round(-1.01*pixmm + printCenter[1])
# maxY = np.round(1.01*pixmm + printCenter[1])


# # print(minX.item(0))
# # print(maxX.item(0))
# # print(minY)
# # print(maxY)


# # print center for testing
# # print(printCenter)

# # Testing on how to assign grayscale to pixels
# x_coords, y_coords, z_coords = new_exo.get_coords()
# elem_conn, num_blk_elems, num_elem_nodes = new_exo.get_elem_connectivity(1)

#     # initiate element centroid array
# element_centroid = [0] * new_exo.num_elems()

# # area calcs function
# def inTriangle(xcoordinates, ycoordinates, p):
#     aT = abs((xcoordinates[0]*(ycoordinates[1] - ycoordinates[2])
#           + xcoordinates[1]*(ycoordinates[2] - ycoordinates[0])
#           + xcoordinates[2]*(ycoordinates[0] - ycoordinates[1]))/2)

#     a1 = abs((p[0]*(ycoordinates[1] - ycoordinates[2])
#           + xcoordinates[1]*(ycoordinates[2] - p[1])
#           + xcoordinates[2]*(p[1] - ycoordinates[1]))/2)
    
#     a2 = abs((xcoordinates[0]*(p[1] - ycoordinates[2])
#           + p[0]*(ycoordinates[2] - ycoordinates[0])
#           + xcoordinates[2]*(ycoordinates[0] - p[1]))/2)
    
#     a3 = abs((xcoordinates[0]*(ycoordinates[1] - p[1])
#           + xcoordinates[1]*(p[1] - ycoordinates[0])
#           + p[0]*(ycoordinates[0] - ycoordinates[1]))/2)

#     # print([aT, round(a1+a2+a3)])
#     # print(round(aT))
#     # note: added the -3 because the areas should be equal, but were coming in with very small differences,
#     # causing whole elements to be left out.
#     # if p[0] in xcoordinates//1:
#     #     node = np.where(xcoordinates//1 == p[0])
#     #     if (ycoordinates//1)[node] == p[1]:
#     #         return 1
#     if (a1 + a2 + a3)//1 > aT:
#         return 0
#     else:
#         return 1


# def inTriangle2(xcoordinates, ycoordinates, p):
#     aT = abs((xcoordinates[0]*(ycoordinates[1] - ycoordinates[2])
#           + xcoordinates[1]*(ycoordinates[2] - ycoordinates[0])
#           + xcoordinates[2]*(ycoordinates[0] - ycoordinates[1]))/2)

#     a1 = abs((p[:,0]*(ycoordinates[1] - ycoordinates[2])
#           + xcoordinates[1]*(ycoordinates[2] - p[:,1])
#           + xcoordinates[2]*(p[:,1] - ycoordinates[1]))/2)
    
#     a2 = abs((xcoordinates[0]*(p[:,1] - ycoordinates[2])
#           + p[:,0]*(ycoordinates[2] - ycoordinates[0])
#           + xcoordinates[2]*(ycoordinates[0] - p[:,1]))/2)
    
#     a3 = abs((xcoordinates[0]*(ycoordinates[1] - p[:,1])
#           + xcoordinates[1]*(p[:,1] - ycoordinates[0])
#           + p[:,0]*(ycoordinates[0] - ycoordinates[1]))/2)
    
#     aTotal = a1 + a2 + a3
#     # print(aTTest)
#     # print(aTotal)
#     # print(len(aTotal))
#     return p[np.where(aTotal-10<=aT//1),:]



# # find the centroid of each element
# # print(new_exo.num_elems())
# for e in range(1): #range(new_exo.num_elems()):
#     print(e)
#     # create an array of the of the nodes that are in element e
#     nodes_in_element = elem_conn[num_elem_nodes*(e):num_elem_nodes*e + num_elem_nodes]
#     # assign the x, y, and z coordinates of each node in element e
#     node_x_coords = x_coords[nodes_in_element-1]
#     node_y_coords = y_coords[nodes_in_element-1]
#     node_z_coords = z_coords[nodes_in_element-1]

#     grayValue = int(round(midGray(props[e][0])))
#     # print(grayValue)

#     minXLocal = min(node_x_coords)*pixmm + printCenter[0]
#     minYLocal = min(node_y_coords)*pixmm + printCenter[1]
#     maxXLocal = max(node_x_coords)*pixmm + printCenter[0]
#     maxYLocal = max(node_y_coords)*pixmm + printCenter[1]

#     # print(minXLocal)
#     # print(maxXLocal)

#     # print(pixmm*node_x_coords + printCenter[0])
#     # print(pixmm*node_y_coords + printCenter[1])
#     xScaled = pixmm*node_x_coords + printCenter[0]
#     yScaled = pixmm*node_y_coords + printCenter[1]



#     # calculate if a pixel is in the element
#     for i in range(int(minXLocal-1),int(maxXLocal+1)):
#         for j in range(int(minYLocal-1),int(maxYLocal+1)):
#             # print([i,j])
#             if inTriangle(xScaled, yScaled, [i,j]) == 1:
#                 buildImage = buildImage.at[j,i].set(grayValue)
#                 # print([i,j,grayValue])




# xFig = np.arange(0, xheight, 1)
# yFig = np.arange(0, yheight, 1)

# fig, ax = plt.subplots()
# meshImage = ax.pcolormesh(xFig, yFig, buildImage)
# fig.colorbar(meshImage)
# ax.set_aspect('equal')
# # scatter to find element nodes for troubleshooting
# # area = [0.01, 0.01, 0.01]
# # plt.scatter(node_x_coords*pixmm + printCenter[0],node_y_coords*pixmm + printCenter[1], s=area)

# plt.savefig('mesh_to_Image_Test_3.png', dpi = 600)
# plt.close()











# new_exo.close()





################################################################################


x = np.array([0.26, 0.58 ,0.61 ])
y = np.array([0.24385, 0.18651 , 0.7231])
z = np.array([0.5, 0.5, 0.5])
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto')
plt.plot(x, y, "ok", label="input point")
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.savefig('linearInterp_1.png', dpi = 600)
