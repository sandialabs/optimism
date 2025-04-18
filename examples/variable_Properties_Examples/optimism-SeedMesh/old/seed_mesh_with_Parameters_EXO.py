import exodus3 as exodus
import numpy as np
import os


# this is just a contrived example of a spatially/temporally
# varying bulk modulus field
def param_function(x, y, z, mult):
    return mult * np.sqrt(x**2 + y**2 + z**2) 


# change these later on
mesh_file = 'cheese.exo'
new_mesh_file = os.path.splitext(mesh_file)[0] + '_with_temperature.exo'

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

# dummy times
times = np.linspace(0.0, 1.0, 100)

# add temperature variable to new_exo
new_exo.set_element_variable_number(2) # 1 for just temperature
new_exo.put_element_variable_name('bulk', 1)
new_exo.put_element_variable_name('shear',2)

new_exo.summarize()

#shearMod = np.linspace(0.0,1.0, new_exo.num_elems())
#bulkMod = np.linspace(5.0, 10.0, new_exo.num_elems())
elem_conn, num_blk_elems, num_elem_nodes = new_exo.get_elem_connectivity(1)
print(elem_conn)
print(new_exo.num_elems())
print(num_elem_nodes)
print(num_blk_elems)


# do some calculation to get the parameters for each element
bulkMod = param_function(x_coords, y_coords, z_coords, 2)
shearMod = param_function(x_coords, y_coords, z_coords, 5)
print(len(bulkMod))


# loop over times
for n, t in enumerate(times):
    # add new time to new_exo
    new_exo.put_time(n + 1, t)
    
    # put that temperature field on new_exo
    new_exo.put_element_variable_values(1, 'bulk', n+1 , bulkMod)
    new_exo.put_element_variable_values(1, 'shear',n+1, shearMod)

# test the assignment of the bulk and shear modulus values
Test_bulk = new_exo.get_element_variable_values(1, 'bulk', 100)
Test_shear = new_exo.get_element_variable_values(1, 'shear',100)
#print(Test_bulk[999])
#print(Test_shear[999])
#print(new_exo.get_coords()[0][0:3])
#print(new_exo.get_coords()[1][0:3])
#print(new_exo.get_coords()[2][0:3])


element_centroid = [0] * new_exo.num_elems()
element_norm = element_centroid[:]
print(len(elem_conn))
print(new_exo.num_elems())
for i in range(new_exo.num_elems()):
    nodes_in_element = elem_conn[num_elem_nodes*(i):num_elem_nodes*i + num_elem_nodes]
    node_x_coords = x_coords[nodes_in_element-1]
    node_y_coords = y_coords[nodes_in_element-1]
    node_z_coords = z_coords[nodes_in_element-1]
    element_centroid[i] = [np.average(node_x_coords), np.average(node_y_coords), np.average(node_z_coords)]
    element_norm[i] = np.linalg.norm(element_centroid[i])

print(element_centroid)
print(element_norm)
print(min(element_norm))
print(max(element_norm))


new_exo.close()
