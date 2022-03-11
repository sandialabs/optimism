from optimism.JaxConfig import *
import csv
from matplotlib import pyplot as plt


alRes = np.load('al_residuals.npz')['data']
altRes = np.load('alt_residuals.npz')['data']


plt.semilogy(altRes, 'b')
plt.semilogy(alRes,'r')
plt.xlabel('iteration')
plt.ylabel('residual norm')
plt.legend(['alternating min', 'trust-region'])
plt.savefig('res_compare.png')



