from matplotlib import pyplot as plt
import numpy as np

plotData = np.load('arch_traction_Fd.npz')

F = plotData['force']
U = plotData['displacement']
plt.plot(U, F)
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.savefig('FD_arch.pdf')
