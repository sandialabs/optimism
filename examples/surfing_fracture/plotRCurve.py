from matplotlib import pyplot as plt
import numpy as np

rCurveData = np.load('r_curve.npz')

a = rCurveData['crackLengthHistory']
J = rCurveData['J']
plt.plot(a-a[0], J, marker='o')
plt.xlabel('Crack extension')
plt.ylabel('Energy release rate (J integral)')
Jmax = np.max(J)
top = 1.1*Jmax if Jmax > 0.0 else 1.0
plt.ylim(bottom=0.0, top=top)
plt.show()
