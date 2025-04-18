import pyvista as pv

plotter = pv.Plotter(notebook=False, off_screen=True)
mesh = pv.read('workdir1/output-040.vtk')
dens = mesh.get_array('element_property_field')
plotter.add_mesh(mesh, 
                 scalars = dens, 
                #  show_edges=True,
                 scalar_bar_args={"title": "Density", "position_x": 0.2,"position_y": 0.},
                 clim = [0,1],
                 cmap = 'coolwarm'
                 )

# print(mesh['element_property_field'])
n_frames = 50
camera_position = [0, 0, 1]
plotter.open_gif("test_GIF_4.gif", fps = 5)
plotter.camera_position = 'xy'
pv.set_plot_theme("paraview")
plotter.write_frame()
# pv.global_theme.colorbar_horizontal.position_x = -0.5
for i in range(1, 50):
    meshTemp = pv.read(f'workdir{i}/output-040.vtk')
    # print(meshTemp.get_array('element_property_field'))
    z = meshTemp.get_array('element_property_field')
    plotter.mesh['element_property_field'][:] = meshTemp.get_array('element_property_field')
    # plotter.update_scalars(z,render=False)
    # print(mesh['element_property_field'])
    plotter.write_frame()

# print(plotter.mesh.scalars)

plotter.close()