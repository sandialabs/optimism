from optimism.JaxConfig import *
import csv
from matplotlib import pyplot as plt


jaxData = np.load('alternating_iters.npz')
plt.plot(jaxData['numAltMinSteps'],'b')

jaxData = np.load('trust_iters.npz')
plt.plot(jaxData['numTrustRegionSolves'],'r')

plt.legend(['alternating min', 'trust-region'])
plt.savefig('iteration_compare.png')


plt.clf()

# timings

jaxData = np.load('alternating_iters.npz')
plt.plot(jaxData['runTimes'],'b')

jaxData = np.load('trust_iters.npz')
plt.plot(jaxData['runTimes'],'r')

plt.ylim([0, 1000])


plt.legend(['alternating min', 'trust-region'])
plt.savefig('time_compare.png')

