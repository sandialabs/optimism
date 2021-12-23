from optimism.JaxConfig import *
import csv
from matplotlib import pyplot as plt


def plot_result(dir, name, signs, ylim=None):
    jaxData = np.load(dir+'/force_disp.npz')

    plt.clf()
    plt.plot(-jaxData['disp'], jaxData['force']*signs[0], 'rv-')
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.savefig(name)    


plot_result('.', 'fd', [1,1])



