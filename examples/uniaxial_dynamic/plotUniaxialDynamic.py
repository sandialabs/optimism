from matplotlib import pyplot as plt
import numpy as np
import UniaxialDynamic

historyData = np.load('history.npz')
u = historyData['displacement']
ke = historyData['kinetic_energy']
se = historyData['strain_energy']
t = historyData['time']


fig, axs = plt.subplots(2, figsize=(12,10))

axs[0].plot(t, u, marker='o', linestyle='--')
axs[0].set(xlabel='time', ylabel='tip displacement')

axs[1].plot(t, ke, 'k-', label='KE')
axs[1].plot(t, se, 'r-', label='SE')
axs[1].plot(t, ke + se, '--', label='TOTAL')
axs[1].set(xlabel='time', ylabel='energy')
axs[1].set_ylim(0.0, None)
axs[1].legend()

plt.tight_layout()

plt.show()
