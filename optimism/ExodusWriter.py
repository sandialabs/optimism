import exodus3 as exodus
import os
from typing import List


def copy_exodus_mesh(exo_file: str, new_exo_file: str) -> None:
    try:    
        os.system('rm -f %s' % new_exo_file)
    except FileNotFoundError:
        pass
    exo = exodus.exodus(exo_file, array_type='numpy')
    exo_copy = exo.copy(new_exo_file)
    exo.close()
    exo_copy.close()


def setup_exodus_database(exo_file: str, 
                          num_node_variables: int, 
                          num_element_variables: int,
                          node_variable_names: List[str], 
                          element_variable_names: List[str]) -> None:

    exo = exodus.exodus(exo_file, 'a', array_type='numpy')
    exo.set_node_variable_number(num_node_variables)
    exo.set_element_variable_number(num_element_variables)

    for n in range(len(node_variable_names)):
        exo.put_node_variable_name(node_variable_names[n], n + 1)

    for n in range(len(element_variable_names)):
        exo.put_element_variable_name(element_variable_names[n], n + 1)

    exo.close()


def write_nodal_variables():
    pass