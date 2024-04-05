from matplotlib import pyplot as plt
import numpy as np

def plot_total_work(fileName, linestyle='-'):
    energies = np.load(fileName)
    totalWork = energies['externalWork']
    time = energies['time']
    plt.plot(time, totalWork, linestyle=linestyle)

def plot_algorithmic_potential(fileName, linestyle='-'):
    energies = np.load(fileName)
    strainEnergy = energies['algorithmicPotential']
    time = energies['time']
    plt.plot(time, strainEnergy, linestyle=linestyle)

def plot_dissipated_energy(fileName, linestyle='-'):
    energies = np.load(fileName)
    dissipatedEnergy = energies['dissipation']
    time = energies['time']
    plt.plot(time, dissipatedEnergy, linestyle=linestyle)

def plot_force_disp(fileName, linestyle='-'):
    histories = np.load(fileName)
    forces = histories['forces']
    disps = histories['disps']
    plt.plot(disps, forces, linestyle=linestyle)

# plot energy histories
energyPrefix = "energy_histories"
forceDispPrefix = "force_disp_histories"
fileType = ".npz"

energyFile = energyPrefix + fileType
plot_total_work(energyFile, '-')
plot_algorithmic_potential(energyFile, ':')
plot_dissipated_energy(energyFile, '--')

plt.xlabel('Time')
plt.ylabel('Energy (mJ)')
plt.legend(["External Work", "Algorithmic Potential Energy", "Dissipated Energy"], loc=0, frameon=True)
plt.savefig('energy_histories.png')

plt.figure()
forceDispFile = forceDispPrefix + fileType
plot_force_disp(forceDispFile, '-')

plt.xlabel('Displacement')
plt.ylabel('Force')
plt.savefig('force_disp.png')
