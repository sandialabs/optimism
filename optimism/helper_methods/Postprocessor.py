# import exodus3 as exodus
import numpy as onp
from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Objective
from optimism import VTKWriter
from optimism.Mechanics import MechanicsFunctions
from optimism.Mesh import Mesh

from typing import Callable


def setup_vtk_writer(base_file_name: str, mesh: Mesh) -> VTKWriter:
    return VTKWriter.VTKWriter(mesh, base_file_name)


def write_standard_fields_static(
    writer: VTKWriter, 
    Uu: np.ndarray,
    p: np.ndarray,
    f_space: FunctionSpace.FunctionSpace,
    create_field: Callable,
    mech_funcs: MechanicsFunctions) -> None:

    U = create_field(Uu, p)
    state = mech_funcs.compute_updated_internal_variables(U, p.state_data)
    p = Objective.param_index_update(p, 1, state)
    writer.add_nodal_field(name='displacement', nodalData=onp.array(U), fieldType=VTKWriter.VTKFieldType.VECTORS)

    _, stresses = mech_funcs.compute_output_energy_densities_and_stresses(U, state)
    cellStresses = FunctionSpace.project_quadrature_field_to_element_field(f_space, stresses)
    writer.add_cell_field(name='stress', cellData=onp.array(cellStresses),
                        fieldType=VTKWriter.VTKFieldType.TENSORS)

    writer.write()
