from optimism.JaxConfig import *
import csv
from matplotlib import pyplot as plt


step = 16

for step in range(1,40):

    firstOrderRes = np.load('contact_residuals_first.'+str(step)+'.npz')['data']
    secondOrderRes = np.load('contact_residuals.'+str(step)+'.npz')['data']

    plt.clf()
    plt.semilogy(firstOrderRes, 'bo--')
    plt.semilogy(secondOrderRes,'ro--')
    plt.xlabel('iteration')
    plt.ylabel('residual norm')
    plt.legend(['standard update', 'super-linear update'])
    plt.savefig('res_compare'+str(step)+'.png')



