from matplotlib import pyplot as plt
import numpy as np
 

loadDisp = np.load('disp_control_response.npz')
 
norm_x = 1.0
norm_y = 1.0
 
F = norm_y*np.abs(loadDisp['force'])
U = norm_x*np.abs(loadDisp['displacement'])
F2 = norm_y*np.abs(loadDisp['targetForces'])
U2 = norm_x*np.abs(loadDisp['targetDisplacements'])
 
plt.plot(U, F, marker='o')
plt.scatter(U2, F2, marker='x', c='r', s=80)
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.legend(["Initial", "Target"], loc=0, frameon=True)
 
plotComparison = True
if plotComparison:
    dispControlData3 = np.load('disp_control_response.npz')
    # dispControlData = np.load('disp_control_response_lbd.npz')
    F3 = norm_y*np.abs(dispControlData3['force'])
    U3 = norm_x*np.abs(dispControlData3['displacement'])
 
    plt.figure()
    plt.plot(U, F, marker='s')
    plt.scatter(U2, F2, marker='x', c='r', s=80)
    plt.plot(U3, F3, marker='x')
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Force (N)')
    plt.legend(["Initial", "Target", "Optimized"], loc=0, frameon=True)
 
# plt.show()
plt.savefig('force_displacement.png')