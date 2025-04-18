********** README **********

This folder uses plato-optimism to calculate the optimal
distribution of density values throughout the mesh.
The StagedMaterialPropertiesOptimization code takes in
a mesh and a target displacement/force array. Then
the mesh is compressed and retracted. If plato is
used, then this simulation is run multiple times until 
the L2 norm difference between the target loads and the 
resulting force-displacement curve is matched as best 
as possible. 

Running the plot_force_displacement_with_target.py will
create a graph with the initial guess (workdir1), the 
final optimization (last workdir), and the target force-
displacements specified in tractionForceFit.py. 

Running mesh_2_Image_GrayScale.py will convert the second-
to-last workdir result vkt to a grayscale image. The code 
allows for changing the "print resolution" (pixels x pixels),
the physical size of the "build plate", the scaling of the
mesh size, and the density-to-grayscale conversion. It 
should be noted that this conversion match the same 
conversion taht is done in the respective material model. 

running the run.sh file will execute all three of these.
