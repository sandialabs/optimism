from matplotlib import pyplot as plt
import numpy as np
 
norm_x = 1.0
norm_y = 1.0



# loadDisp = np.load('disp_control_response_050.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='1')


loadDisp = np.load('force_control_response.npz')
F = norm_y*np.abs(loadDisp['force'])
U = norm_x*np.abs(loadDisp['displacement'])
plt.plot(U, F, marker='3')
# loadDisp = np.load('force_control_response_dt_005_H.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='4')

# loadDisp = np.load('force_control_response_dt_005.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='3')

# loadDisp = np.load('force_control_response_dt_05.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='3')

# loadDisp = np.load('force_control_response_dt_1.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='2')

# loadDisp = np.load('force_control_response_dt_2.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='1')

# loadDisp = np.load('disp_control_response_040.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='4')

# loadDisp = np.load('disp_control_response_050.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='1')

# loadDisp = np.load('disp_control_response_100.npz')
# F = norm_y*np.abs(loadDisp['force'])
# U = norm_x*np.abs(loadDisp['displacement'])
# plt.plot(U, F, marker='x')

compareOptimized = True
if compareOptimized:
    

    F2 = norm_y*np.abs(loadDisp['targetForces'])
    U2 = norm_x*np.abs(loadDisp['targetDisplacements'])
    plt.scatter(U2, F2, marker='x', c='r', s=80)
    plt.legend(["Response","Target"], loc=0, frameon=True)
    # plt.legend(["dt 005 H H", "dt 005 H","dt 005", "dt 025","dt 05","dt 1",  "dt 2"], loc=0, frameon=True)
else:
    plt.legend(["Initial", "Target"], loc=0, frameon=True)

plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
 
plt.savefig('force_displacement.png')
