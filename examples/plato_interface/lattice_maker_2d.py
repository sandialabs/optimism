from typing import Optional
import cubit3 as cubit


def lattice_maker_2d(
    n_across: int,
    n_layers: int,
    spacing: float,
    layer_height: float,
    filament_diameter: float,
    element_size: Optional[float] = None,
    mesh_file_name: Optional[str] = 'mesh.g',
) -> None:
    
    length = (n_across - 1) * (spacing + 1.) * filament_diameter + filament_diameter
    if element_size is None:
        element_size = filament_diameter / 5.

    cubit.init(['cubit',
                '-batch',
                '-nographics',
                '-nojournal',
                '-noecho',
                '-information',
                'off'])
    cubit.reset()

    surface = 0
    for layer in range(n_layers):
        if layer % 2 == 0:
            cubit.cmd('create surface rectangle width %s height %s zplane' % (length, filament_diameter))
            surface = surface + 1
            cubit.cmd('move surface %s x %s y %s' % (surface, length / 2, filament_diameter / 2))
            cubit.cmd('move surface %s x %s y %s' % (surface, 0., layer * layer_height))
        else:
            for n in range(n_across):
                cubit.cmd('create surface circle radius %s zplane' % (filament_diameter / 2))
                surface = surface + 1
                cubit.cmd('move surface %s x %s y %s' % (surface, filament_diameter / 2, filament_diameter / 2))
                cubit.cmd('move surface %s x %s y %s' % (surface, 0., layer * layer_height))
                cubit.cmd('move surface %s x %s y %s' % (surface, n * (spacing + 1.) * filament_diameter, 0.))
                # n = n + 1

    cubit.cmd('unite body all')
    surf = cubit.get_last_id('surface')

    # get a bounding box
    bb = cubit.get_bounding_box('surface', surf)
    y_min, y_max = bb[3], bb[4]

    cubit.cmd('surface %s scheme trimesh' % surf)
    cubit.cmd('surface %s size %s' % (surf, element_size))
    cubit.cmd('mesh surface %s' % surf)
    cubit.cmd('block 1 add surface %s' % surf)
    cubit.cmd('block 1 element type tri6')
    cubit.cmd('sideset 1 add curve with y_coord < %s' % (y_min + 0.000001))
    cubit.cmd('sideset 2 add curve with y_coord > %s' % (y_max - 0.000001))
    cubit.cmd('sideset 1 name "sset_bottom"')
    cubit.cmd('sideset 2 name "sset_top"')
    cubit.cmd('nodeset 1 add curve in sideset 1')
    cubit.cmd('nodeset 2 add curve in sideset 2')
    cubit.cmd('nodeset 1 name "nset_bottom"')
    cubit.cmd('nodeset 2 name "nset_top"')
    cubit.cmd('save as "%s" overwrite' % 'temp.cub')
    cubit.cmd('export genesis "%s" overwrite' % mesh_file_name)


if __name__ == '__main__':
    n_across = 5
    n_layers = 19
    spacing = 5
    layer_height = 0.3526
    filament_diameter = 0.41

    lattice_maker_2d(n_across, n_layers, spacing, layer_height, filament_diameter)
