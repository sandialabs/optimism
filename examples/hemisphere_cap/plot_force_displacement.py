from matplotlib import pyplot as plt
import numpy as np

forceControlData = np.load('force_control_response.npz')

F = forceControlData['force']
U = forceControlData['displacement']
plt.plot(U, F, marker='o')
plt.xlabel('Displacement')
plt.ylabel('Force')

plotComparison = True
if plotComparison:
    dispControlData = np.load('disp_control_response.npz')
    F2 = dispControlData['force']
    U2 = dispControlData['displacement']
    plt.figure()
    plt.plot(U, F, marker='s')
    plt.plot(U2, F2, marker='o')
    plt.xlabel('Displacement')
    plt.ylabel('Force')

plt.show()

#plt.savefig('FD_arch.pdf')
