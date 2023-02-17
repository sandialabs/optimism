import exodus3 as exodus
import jax.numpy as np
import numpy as onp
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
                          element_variable_names: List[str]) -> exodus.exodus:

    assert num_node_variables == len(node_variable_names), \
        'Number of node variables needs to match the number of node variable names!'
    assert num_element_variables == len(element_variable_names), \
        'Number of element variables needs to match the number of element variable names!'

    exo = exodus.exodus(exo_file, 'a', array_type='numpy')
    exo.set_node_variable_number(num_node_variables)
    exo.set_element_variable_number(num_element_variables)

    for n in range(len(node_variable_names)):
        exo.put_node_variable_name(node_variable_names[n], n + 1)

    for n in range(len(element_variable_names)):
        exo.put_element_variable_name(element_variable_names[n], n + 1)

    return exo


def write_exodus_nodal_outputs(exo: exodus.exodus,
                               node_variable_names: List[str],
                               node_variable_values: List[str],
                               time_step: int) -> None:
    
    assert len(node_variable_names) == len(node_variable_values), \
        'Number of node variable names needs to match the number of node variable value arrays!'
    
    for n in range(len(node_variable_names)):
        exo.put_node_variable_values(node_variable_names[n], time_step, onp.array(node_variable_values[n]))

    
def write_exodus_element_outputs(exo: exodus.exodus,
                                 element_variable_names: List[np.ndarray],
                                 element_variable_values: List[np.ndarray],
                                 time_step: int, block_id: int) -> None:
    assert len(element_variable_names) == len(element_variable_values), \
        'Number of element variable names needs to match the number of element variable value arrays!'

    for n in range(len(element_variable_names)):
        exo.put_element_variable_values(block_id, element_variable_names[n], time_step, onp.array(element_variable_values[n]))
