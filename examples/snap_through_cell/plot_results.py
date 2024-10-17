from matplotlib import pyplot as plt
import numpy as np

# Plotting settings
legendFontSize = 14
axisFontSize = 14
lineWidth = 3.0

def plot_total_work(fileName, linestyle='-'):
    energies = np.load(fileName)
    totalWork = energies['totalWork']
    if 'time' in energies:
        time = np.cumsum(energies['time'])
        plt.plot(time, totalWork, linestyle=linestyle, linewidth=lineWidth)
    else:
        plt.plot(totalWork, linestyle=linestyle, linewidth=lineWidth)

def plot_strain_energy(fileName, linestyle='-'):
    energies = np.load(fileName)
    strainEnergy = energies['strainEnergy']
    if 'time' in energies:
        time = np.cumsum(energies['time'])
        plt.plot(time, strainEnergy, linestyle=linestyle, linewidth=lineWidth)
    else:
        plt.plot(strainEnergy, linestyle=linestyle, linewidth=lineWidth)

def plot_dissipated_energy(fileName, linestyle='-'):
    energies = np.load(fileName)
    dissipatedEnergy = energies['dissipatedEnergy']
    if 'time' in energies:
        time = np.cumsum(energies['time'])
        plt.plot(time, dissipatedEnergy, linestyle=linestyle, linewidth=lineWidth)
    else:
        plt.plot(dissipatedEnergy, linestyle=linestyle, linewidth=lineWidth)

def plot_force_disp(fileName, linestyle='-'):
    norm_x = -1.0
    norm_y = -1.0
    histories = np.load(fileName)
    forces = norm_x*histories['force']
    disps = norm_y*histories['displacement']
    plt.plot(disps, forces, linestyle=linestyle, linewidth=lineWidth)

# plot files
energyPrefix = "energy_histories"
forceDispPrefix = "force_control_response"
fileType = ".npz"

# plot force-displacement
forceDispFile = forceDispPrefix + fileType
plot_force_disp(forceDispFile, '-')

plt.xlabel('Displacement (mm)', fontsize=axisFontSize)
plt.ylabel('Force (N)', fontsize=axisFontSize)
plt.xticks(fontsize=axisFontSize)
plt.yticks(fontsize=axisFontSize)
plt.savefig('force_disp.png')

# plot energy histories
plt.figure()

energyFile = energyPrefix + fileType
plot_total_work(energyFile, '-')
plot_strain_energy(energyFile, ':')
plot_dissipated_energy(energyFile, '--')

if 'time' in np.load(energyFile):
    plt.xlabel('Time(s)', fontsize=axisFontSize)
else:
    plt.xlabel('Step', fontsize=axisFontSize)
plt.ylabel('Energy (mJ)', fontsize=axisFontSize)
plt.xticks(fontsize=axisFontSize)
plt.yticks(fontsize=axisFontSize)
plt.legend(["External Work", "Strain Energy", "Dissipated Energy"], loc=0, frameon=True, fontsize=legendFontSize)
plt.savefig('energy_histories.png')
