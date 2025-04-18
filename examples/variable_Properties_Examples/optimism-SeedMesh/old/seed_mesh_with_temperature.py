import exodus3 as exodus
import numpy as np
import os


# create an arbitrary parameter function. This will need to be defined by the user for different cases
# this example function assigns linearly increasing parameters based on the distance from the origin 
#   and a multiplier 
def param_function(x, y, z, mult):
    param = [0]*len(x)
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        z_coord = z[i]
        param[i] =  mult * np.sqrt(x_coord**2 + y_coord**2 + z_coord**2) 
    return param

def add_properties_to_mesh(mesh_file, block_id, property_names, property_funcs, property_params):

    """"
    Description: add any number of properties to each element in a mesh, as defined by a property function.
     
    mesh_file: the mesh that you will be assigning to, confirmed works with .g and .exo
    block_id: the block(s) that will have parameters assigned, currently set to work with only 1 block. 
    property_names: list of the property names that will be assigned, ex: ['bulk', 'shear']
    property_funcs: a function that will be used to define the parameter magnitudes. Currently uses the 
      geometric location of the elements to determine the magnitude of an element parameters.
    property_params: this can be used to pass known values to the property_funcs.
    """

    # create a new mesh file copied from the input file
    new_mesh_file = os.path.splitext(mesh_file)[0] + '_with_parameters.exo'
 
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

    # assign the property values to each property in the elements
    for i in range(len(property_names)):
        new_exo.put_element_variable_values(block_id, property_names[i], 1, property_funcs(element_x, element_y, element_z, property_params[i]))


    # close the .exo file MUST HAVE THIS 
    new_exo.close()


# test the code
add_properties_to_mesh('ellipse_test.exo', 1, ['bulk','shear'], param_function, [1, 9])
