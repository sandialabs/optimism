from optimism.JaxConfig import *
import csv
from matplotlib import pyplot as plt

import numpy as np

steps = 16
fileName = 'fd'


names = np.arange(steps)
names = [str(n).zfill(2) for n in names]

colors = ['r','b','g','c','k','m']

def plot_result(dir, signs, step):
    jaxData = np.load(dir+'/arch-'+str(step).zfill(2)+'.npz')
    plt.plot(-jaxData['disp'], jaxData['force']*signs[0], colors[step%len(colors)])

    
plt.clf()

for step in range(steps):
    plot_result('.', [-1,1], step)

plt.xlabel('Displacement')
plt.ylabel('Force')
#plt.legend(names)
plt.savefig(fileName)    

