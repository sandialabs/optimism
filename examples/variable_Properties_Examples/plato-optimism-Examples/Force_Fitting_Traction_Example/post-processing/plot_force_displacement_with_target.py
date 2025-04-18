from matplotlib import pyplot as plt
import numpy as np
import glob
import os

 # normalize value, kept at 1 currently
norm_x = 1.0
norm_y = 1.0

# function that creates a list of all directories
# in a given directory that matches the given pattern
def get_workdirs(directory,pattern):
    return glob.glob(os.path.join(directory,pattern))

# create a list of all workdir in the current directory
workdirs = get_workdirs('./','workdir*')

# plot the initial guess, assumed to be the first workdir
loadDisp = np.load(workdirs[0] + '/force_control_response.npz')
F = norm_y*np.abs(loadDisp['force'])
U = norm_x*np.abs(loadDisp['displacement'])
plt.plot(U, F, marker='o')


# plot the last/optimized guess, assumed to be the
# second-to-last workdir. This is due to some instances
# of plato solutions not solving, so this ensures that the
# last converged workdir is used. If it converges, the -2 
# should be changed to -1. 
loadDisp = np.load(workdirs[-2] + '/force_control_response.npz')
# loadDisp = np.load('workdir1/force_control_response.npz')
F = norm_y*np.abs(loadDisp['force'])
U = norm_x*np.abs(loadDisp['displacement'])
plt.plot(U, F, marker='3')

# used to toggle plotting the target force-displacement 
# points 
compareTarget = True

# plot the force-displacement target
if compareTarget:
    F2 = norm_y*np.abs(loadDisp['targetForces'])
    U2 = norm_x*np.abs(loadDisp['targetDisplacements'])
    plt.scatter(U2, F2, marker='x', c='r', s=80)
    plt.legend(["Initial", "Optimized", "Target"], loc=0, frameon=True)
else:
    plt.legend(["Initial", "Optimized"], loc=0, frameon=True)

# plot labels
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
 
# save the plot
plt.savefig('./Images/force_displacement.png')
