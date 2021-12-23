from matplotlib import pyplot as plt
import numpy as np

rCurveData = np.load('force-displacement.npz')

d = rCurveData['displacement']
F = rCurveData['force']
plt.plot(d, F, marker='o')
plt.xlabel('Applied displacement')
plt.ylabel('Force')
plt.show()
