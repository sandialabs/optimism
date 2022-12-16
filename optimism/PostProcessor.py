from optimism import VTKWriter

from typing import Callable
from typing import List
from typing import Optional

def setup_vtk_post_processor(base_file_name: str, mesh: any) -> VTKWriter.VTKWriter:
    writer = VTKWriter.VTKWriter(mesh, base_file_name)
    return writer


def write_nodal_field_to_vtk(
    writer: VTKWriter.VTKWriter,
    field_name: str,
    field: any,
    params: any,
    create_field_method: Callable) -> None:

    U = create_field_method(field, params)
    writer.add_nodal_field(name=field_name, nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)


# write method that will output all requested nodal/element variables based
# on simple input and maybe some callable methods to pass around
#