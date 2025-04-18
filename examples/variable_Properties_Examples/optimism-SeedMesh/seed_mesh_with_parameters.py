import exodus3 as exodus
import numpy as np
import os
import scipy as sp
import scipy.special as special
# import matplotlib.pyplot as plt

# At the very bottom of the code is the use of the function
# add_properties_to_mesh(). in that fuction, list the exo file you want
# to seed, the pattern function for assigning the values, and the 
# range of values you want in the mesh. This is in the form of the 
# average value and the deviation (+-). See function description for
# each input option. 


# convert the light dosage to an elastic modulus
def dosage_2_Modulus(dose):
    constProps = {'Ec': 1.059, # MPa
              'b': 5.248, # unitless
              'p_gel': 0.12, # unitless
              'Ed': 3.321, 
              'Er': 18959, 
              'R': 8.314, 
              'g1': 109603, # unitless
              'g2': 722.2, # unitless
              'xi': 3.73,# unitless
              'C1': 61000, # unitless
              'C2': 511.792, # K
              'rmTemp': 100, # C
              'tau': 0.001, # s
              'K': 0.01
              }
    p = 1 - np.exp(-constProps['K']*dose)
    E = (constProps['Ec'] * np.exp(constProps['b'] * (p - constProps['p_gel']))) + constProps['Ed']
    return E

# Calculate the distance from an element to the centroid of the mesh
def element_centroid_distance(x, y, z):
    elemCent = [0]*len(x)
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        z_coord = z[i]
        #param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)/10 
        elemCent[i] =  np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
    return elemCent


# Define the distribution of the parameters throughout the model.
# key values that are passed are the x, y, and z coordinates of the 
#   elements in the mesh, the minimum and maximum values of the euclidean
#   distances from the centroid to the elements (note: this should be phased
#   out). Value represents median property value, and mult 

# linearly increase the param with respect to the radial distance from the centroid
def param_function_radial_increasing(x, y, z,  minimum, maximum, value, mult):
    '''
    Description: create a parameter array whose values are linearly increasing as the 
        distance from the element to the centroid increases. Allows for the variation
        (+- some value) to be set. 

    value: the parameter value around which the param will vary
    min: the minimum distance from origin
    max: the maximum distance from origin
    '''
    param = [0]*len(x)
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        z_coord = z[i]
        A = (-1/minimum) - ((1+(maximum/minimum))/(minimum-maximum))
        B = (1+(maximum/minimum))/(1-(maximum/minimum))
        #param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)/10 
        param[i] =  value + mult*((A * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)) + B)
    return param

# linearly decrease the param with respect to the radial distance from the centroid
def param_function_radial_decreasing(x, y, z, minimum, maximum, value, mult):
    param = [0]*len(x)
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        z_coord = z[i]
        A = (-1/minimum) - ((1+(maximum/minimum))/(minimum-maximum))
        B = (1+(maximum/minimum))/(1-(maximum/minimum))
        #param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)/10 
        param[i] =  value - mult*((A * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)) + B)
    return param

# keep a param constant over the whole mesh
def param_function_const(x,y,z,minimum, maximum, value, mult):
    param = [0]*len(x)
    for i in range(len(x)):
        param[i] = value
    return param

# linearly increase the param with respect to the y coordinates
def param_function_linear_y_increasing(x, y, z, minimum, maximum, value, mult):
    param = [0]*len(x)
    minY = min(y)
    maxY = max(y)
    for i in range(len(x)):
        #x_coord = x[i]
        y_coord = y[i]
        #z_coord = z[i]
        A = (-1/minY) - ((1+(maxY/minY))/(minY-maxY))
        B = (1+(maxY/minY))/(1-(maxY/minY))
        #param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)/10 
        param[i] =  value + mult*((A * y_coord) + B)
        print([i, param[i]])
    # print(np.shape(param))
    return param

# linearly decrease the param with respect to the y coordinates
def param_function_linear_y_decreasing(x, y, z, minimum, maximum, value, mult):
    param = [0]*len(x)
    minY = min(y)
    maxY = max(y)
    for i in range(len(x)):
        #x_coord = x[i]
        y_coord = y[i]
        #z_coord = z[i]
        A = (-1/minY) - ((1+(maxY/minY))/(minY-maxY))
        B = (1+(maxY/minY))/(1-(maxY/minY))
        #param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)/10 
        param[i] =  value - mult*((A * y_coord) + B)
    return param

# a bessel function 
def drumhead_height(n, k, distance, angle, t):
   kth_zero = special.jn_zeros(n, k)[-1]
   return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)

# calculate the parameters using a bessel function of the first kind
def param_function_bessel_1(x,y,z,minimum,maximum, value, mult):
    #x = np.linspace(-1,1,100)
    #y = x[:]
    # xv, yv = np.meshgrid(x,y)
    bessel = [0]*len(x)
    param = bessel[:]
    denser = 1 # default to 1. more zeros in bessel >1, few <1
    xshift = 0
    yshift = 0
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        bessel[i] = drumhead_height(2,2,denser*np.sqrt((x_coord-xshift)**2 + (y_coord-yshift)**2), np.arctan2((y_coord-yshift),(x_coord-xshift)),0)
    # bessel = np.array([drumhead_height(2, 2, np.sqrt(X**2 + y**2), np.atan2(y,X), 0) for X in x])
    bess_min = np.min(bessel)
    bess_max = np.max(bessel)

    A = (-1/bess_min) - ((1+(bess_max/bess_min))/(bess_min-bess_max))
    B = (1+(bess_max/bess_min))/(1-(bess_max/bess_min))
    # print(bessel)
    for i in range(len(x)):
        param[i] = value - mult*((A * bessel[i]) + B)

    return param

# this flips the x and y axes, specifically in the arctan2 function, of the bessel function distribution
def param_function_bessel_2(x,y,z,minimum,maximum, value, mult):
    #x = np.linspace(-1,1,100)
    #y = x[:]
    # xv, yv = np.meshgrid(x,y)
    bessel = [0]*len(x)
    param = bessel[:]
    denser = 1.0
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        bessel[i] = drumhead_height(1,2,denser*np.sqrt(x_coord**2 + y_coord**2), np.arctan2(x_coord,y_coord),0)
    # bessel = np.array([drumhead_height(2, 2, np.sqrt(X**2 + y**2), np.atan2(y,X), 0) for X in x])
    bess_min = np.min(bessel)
    bess_max = np.max(bessel)

    A = (-1/bess_min) - ((1+(bess_max/bess_min))/(bess_min-bess_max))
    B = (1+(bess_max/bess_min))/(1-(bess_max/bess_min))
    # print(bessel)
    for i in range(len(x)):
        param[i] = value - mult*((A * bessel[i]) + B)

    return param

# calculate the sinusoidal distribution of a parameter
def param_function_sinusoidal(x,y,z,minimum,maximum, value, mult):
    
    param = [0]*len(x) # initialize param array
    xshift = 0 #shift in the x direction
    yshift = 0 # shift in the y direction
    wL = 0.75 # wavelength of the parameters
    km = 2*np.pi/wL # calc constant
    shiftMult = -1.5
    for i in range(len(x)):
        x_coord = x[i] # assign x coordinate
        y_coord = y[i] # assign y coordinate
        param[i] = value + mult*np.cos(km*(x_coord + xshift) + shiftMult*(y_coord + yshift)) # sinusoidal parameters

    return param

# define a horizontal line for the parameters
def param_function_horiz_line(x,y,z,minimum,maximum,value,mult):
    param = [0]*len(x)
    # print(value)
    # print(mult)
    xmin = min(x)
    xmax = max(x)
    for i in range(len(x)):
        y_coord = y[i]
        x_coord = x[i]
        
        if x_coord < xmax and x_coord > xmin:
            if y_coord < 0.5 and y_coord > 0.25:
                param[i] = value - mult
            else:
                param[i] = value + mult
        else:
            param[i] = value + mult
    return param


# calculate the distance from each element to a series of connected straight lines DO NOT USE
# def param_function_zig_zag(x,y,z,minimum,maximum,value,mult):
    param = [0]*len(x)
    xZig = [0, -1, 2]
    yZig = [2, -1, 0]
    a =[0]*(len(xZig)-1)
    b = a[:]
    c = a[:]
    for k in range(len(xZig)-1):
        a[k] = yZig[k+1] - yZig[k]
        b[k] = xZig[k] - xZig[k+1]
        c[k] = a[k]*xZig[k] + b[k]*yZig[k]
    for i in range(len(x)):
        d = [0]*len(a)
        for j in range(len(a)):
            d[j] = abs(a[j]*x[i] + b[j]*y[i] - c[j])/np.sqrt(a[j]**2 + b[j]**2)
        if any(ele < 0.1 for ele in d):
            param[i] = value - mult
        else:
            param[i] = value + mult
    return param

# zig zag with softer edges:
def param_function_zig_zag_smooth(x,y,z,minimum,maximum,value,mult):
    param = [0]*len(x)
    # xZig = [0.75,-0.25,0.75,-0.5]
    # yZig = [1.25,0.25,0.25,-2]
    # xZig = [-1.5,-0.5,-0.75,0.5,-0.1,0.9,0.5]
    # yZig = [0.1,0.75,-0.3,0.6,-0.8,-0.75,-1.5]
    # xZig = [0,0.5,0,-0.5,0]
    # yZig = [0.75,0,-0.75,0,0.75]
    xZig = [-2, 1, 1, 4,  4,  3,  3,  2, 2, 1,  2, -2, -1, -2, -2, -3, -3, -4, -4, -1, -1, -2, -2]
    yZig = [ 5, 5, 3, 3, -3, -3, -1, -1, 1, 1, -5, -5,  1,  1, -1, -1, -3, -3,  3,  3,  4,  4,  5]
    a =[0]*(len(xZig)-1)
    b = a[:]
    c = a[:]
    for k in range(len(xZig)-1):
        a[k] = yZig[k+1] - yZig[k]
        b[k] = xZig[k] - xZig[k+1]
        c[k] = a[k]*xZig[k] + b[k]*yZig[k]
    # print(b)
    for i in range(len(x)):
        #d = [0]*len(a)
        #phi = d[:]
        validD = []
        for j in range(len(a)):
            d = abs(a[j]*x[i] + b[j]*y[i] - c[j])/np.sqrt(a[j]**2 + b[j]**2)
            # print(j)
            # print(d)
            # print([j, b[j], d])
            if b[j] == 0: # if the line made by two vertices is verticle, then:
                if y[i] > min([yZig[j],yZig[j+1]]) and y[i] < max([yZig[j],yZig[j+1]]): # if the point is between the two points in the y direction, 
                    validD.append(d)
                    # print(validD[j])
            
            else:

                phi = (x[i] + (a[j]*c[j]/(b[j]**2)) - (a[j]*y[i]/b[j]))/(((a[j]**2)/(b[j]**2))+1)
                # print(phi)
                # print([xZig[j], xZig[j+1]])
                if phi > min([xZig[j],xZig[j+1]]) and phi < max([xZig[j],xZig[j+1]]):
                    # print(phi)
                    # print([xZig[j], xZig[j+1]])
                    validD.append(d)  
                    # print(validD[j])         
                else:
                    dPoint = [0,0]
                    # print([xZig[j], xZig[j+1]])
                    for l in range(2):
                        # print(xZig[l+j])
                        dPoint[l] = np.sqrt(((xZig[l+j]-x[i])**2) + ((yZig[l+j] - y[i])**2))
                    validD.append(min(dPoint))
                    # print(validD[j])
                # print(dPoint)
            # print(validD)
            # print([j, validD[j]])
            # print(validD)
        # print(validD)
        minD = min(validD)
        # print(minD)
        param[i] = value - mult + (2 * mult * (np.sqrt((minD))/np.sqrt(np.sqrt(2)*2)))
        # print(param[i])
    return param

# function to add parameters to mesh
def add_properties_to_mesh(mesh_file, block_id, property_names, property_funcs, constant_params, property_params, multiplier):

    """"
    Description: add any number of properties to each element in a mesh, as defined by a property function.
     
    mesh_file: the mesh that you will be assigning to, confirmed works with .g and .exo
    block_id: the block(s) that will have parameters assigned, currently set to work with only 1 block. 
    property_names: list of the property names that will be assigned, ex: ['bulk', 'shear']
    property_funcs: a function that will be used to define the parameter magnitudes. Currently uses the 
      geometric location of the elements to determine the magnitude of an element parameters.
    constant_params: array to specify if any of the parameters are constant over all elements. 0 is variable, 1 is constant
    property_params: this can be used to pass known values to the property_funcs.
    """

    # create a new mesh file copied from the input file
    new_mesh_file = os.path.splitext(mesh_file)[0] + '_Seeded.exo'
 
    # delete exisiting file if it's there
    if os.path.isfile(new_mesh_file):
        os.remove(new_mesh_file)

	# open mesh file and copy it, exodus.py has some other flakey copy methods
    exo = exodus.exodus(mesh_file, mode='r', array_type='numpy')
    new_exo = exo.copy(new_mesh_file) # returns new opened exodus database
    exo.close()
    new_exo.close()

	# we have to re-open because exodus.py is weird and not intuitive
    new_exo = exodus.exodus(new_mesh_file, mode='a', array_type='numpy')

	# useful method below for printing stuff out
    new_exo.summarize()

	# get coordinates
    x_coords, y_coords, z_coords = new_exo.get_coords()

    # add parameter variables to new_exo
    new_exo.set_element_variable_number(len(property_names)) 
    for n, name in enumerate(property_names):	
        new_exo.put_element_variable_name(name, n + 1)

    new_exo.summarize()

    # this sets the timestep to 1, and the time to zero, for the parameter assignment [MAY NOT BE NECESSARY]
    #new_exo.put_time(1, 0.0)

    elem_conn, num_blk_elems, num_elem_nodes = new_exo.get_elem_connectivity(1)

    # initiate element centroid array
    element_centroid = [0] * new_exo.num_elems()

    # find the centroid of each element
    for i in range(new_exo.num_elems()):
        # create an array of the of the nodes that are in element i
        nodes_in_element = elem_conn[num_elem_nodes*(i):num_elem_nodes*i + num_elem_nodes]

        # assign the x, y, and z coordinates of each node in element i
        node_x_coords = x_coords[nodes_in_element-1]
        node_y_coords = y_coords[nodes_in_element-1]
        node_z_coords = z_coords[nodes_in_element-1]

        # find the centroid of element i, assign the coordinates to element_centroid
        element_centroid[i] = [np.average(node_x_coords), np.average(node_y_coords), np.average(node_z_coords)]


    # create an array of the cartesian coordinates for the elements in the mesh
    element_x = [x[0] for x in element_centroid]
    element_y = [x[1] for x in element_centroid]
    element_z = [x[2] for x in element_centroid]

    # calculate the distances from the centroid to each element
    element_centroid_norms = element_centroid_distance(element_x, element_y, element_z)

    maxNorm = max(element_centroid_norms)
    minNorm = min(element_centroid_norms)


    # assign the property values to each property in the elements
    for i in range(len(property_names)):
        if constant_params[i] < 1:
            new_exo.put_element_variable_values(block_id, property_names[i], 1, property_funcs[i](element_x, element_y, element_z,minNorm, maxNorm, property_params[i], multiplier[i]))
            #print((property_funcs[0](element_x, element_y, element_z,minNorm, maxNorm, property_params[i], multiplier[i])))
            #print(min(property_funcs[0](element_x, element_y, element_z,minNorm, maxNorm, property_params[i], multiplier[i])))
            #print(max(property_funcs[0](element_x, element_y, element_z,minNorm, maxNorm, property_params[i], multiplier[i])))
        else:
            new_exo.put_element_variable_values(block_id, property_names[i], 1, property_funcs[i](element_x, element_y, element_z,  minNorm, maxNorm, property_params[i], multiplier[i]))
            

    # close the .exo file MUST HAVE THIS 
    new_exo.close()

# test the code
#add_properties_to_mesh('ellipse_test.exo', 1, ['bulk','shear'], [param_function_radial_decreasing, param_function_const], [0, 1], [50.5, 0.48], [49.5, 0])
#add_properties_to_mesh('ellipse_test.exo', 1, ['light_intensity'], param_function, [1.])
# add_properties_to_mesh('./EXO_files/ellipse_test.exo', 1, ['Youngs_Modulus', 'nu'], [param_function_linear_y_decreasing, param_function_const], [0, 1], [1010, 0.48], [990, 0])
add_properties_to_mesh('./EXO_files/ellipse_test.exo', 1, ['density', 'nu'], [param_function_linear_y_decreasing, param_function_const], [0, 1], [0.5, 0.48], [0.5, 0])
# add_properties_to_mesh('beamTest_1.exo', 1, ['light_intensity'], [param_function_horiz_line, param_function_const], [1], [0.5], [0])

# functional
# add_properties_to_mesh('./read_variable_material_property_test.exo', 1, ['light_dose', 'nu'], [param_function_linear_y_increasing, param_function_const], [0, 1], [260, 0.48], [240, 0])
