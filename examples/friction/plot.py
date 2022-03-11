from optimism.JaxConfig import *
import csv
from matplotlib import pyplot as plt


def plot_result(dir, name):
    jaxData = np.load(dir+'/force_disp.npz')

    plt.clf()
    plt.plot(jaxData['force'], -jaxData['disp'], 'bv-')
    plt.ylabel('Displacement')
    plt.xlabel('Force')
    plt.savefig(name)    


plot_result('.', 'force_vs_disp')



