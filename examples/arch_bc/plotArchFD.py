from matplotlib import pyplot as plt
import numpy as np

plotData = np.load('arch_bc_Fd.npz')

F = plotData['force']
U = plotData['displacement']
E = plotData['energy']


plt.rcParams["mathtext.fontset"]='cm'
plt.rcParams['font.size'] = '18'

# find snap-through point
iC = np.argmax(F[:F.size//2])
FC = F[iC]
UC = U[iC]

# part way to snap-through
iB = iC//3
FB = F[iB]
UB = U[iB]

print('index of point B', iB)

# minimum force in snap-through region
iD = iC + np.argmin(F[iC:])
FD = F[iD]
UD = U[iD]

# find point E, which has force equal to snap-through point C
# (find closest simulated point to this)
iE = iD + np.argmin(np.abs(F[iD:] - FC))
FE = F[iE]
UE = U[iE]

print("UB = ", UB, "FB = ", FB)
print("UC = ", UC, "FC = ", FC)
print("UD = ", UD, "FD = ", FD)
print("UE = ", UE, "FE = ", FE)

ULabeled = np.array([U[0], UB, UC, UD, UE, U[-1]])
FLabeled = np.array([F[0], FB, FC, FD, FE, F[-1]])
ELabeled = E[[0,iB,iC,iD,iE,-1]]
labels = ['a', 'b', 'c', 'd', 'e', 'f']


def label_points(x, y):
    for i,lbl in enumerate(labels):
        plt.annotate(lbl,
                     (x[i], y[i]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')


fig = plt.figure()
plt.plot(U, F) # marker='o'
plt.scatter(ULabeled, FLabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Force')
label_points(ULabeled, FLabeled)
bottom, top = plt.ylim()
plt.ylim(top=bottom+1.1*(top-bottom))
plt.tight_layout()
plt.savefig('arch_FD.pdf')


# synchronize ylimits for all energy plots
yMax = 0.035
yMin = -0.025

fig = plt.figure()
plt.plot(U, E)
plt.scatter(ULabeled, ELabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Energy', title=r'$f = f_a = 0$')
bottom, top = plt.ylim()
plt.ylim(bottom=yMin, top=yMax)
label_points(ULabeled, ELabeled)
plt.tight_layout()
ax.set_aspect(1 / ax.get_data_ratio())
plt.savefig('arch_energy_state_A.pdf')

fig = plt.figure()
plt.plot(U, E - FB*U)
plt.scatter(ULabeled, ELabeled - FB*ULabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Energy', title=r'$f = f_b$')
bottom, top = plt.ylim()
plt.ylim(bottom=yMin, top=yMax)
label_points(ULabeled, ELabeled - FB*ULabeled)
plt.tight_layout()
ax.set_aspect(1 / ax.get_data_ratio())
plt.savefig('arch_energy_state_B.pdf')

fig = plt.figure()
plt.plot(U, E - FC*U)
plt.scatter(ULabeled, ELabeled - FC*ULabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Energy', title=r'$f = f_c = f_e$')
bottom, top = plt.ylim()
plt.ylim(bottom=yMin, top=yMax)
label_points(ULabeled, ELabeled - FC*ULabeled)
plt.tight_layout()
ax.set_aspect(1 / ax.get_data_ratio())
plt.savefig('arch_energy_state_C.pdf')


fig = plt.figure()
plt.plot(U, E - FD*U)
plt.scatter(ULabeled, ELabeled - FD*ULabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Energy', title=r'$f = f_d$')
bottom, top = plt.ylim()
plt.ylim(bottom=yMin, top=yMax)
label_points(ULabeled, ELabeled - FD*ULabeled)
plt.tight_layout()
ax.set_aspect(1 / ax.get_data_ratio())
plt.savefig('arch_energy_state_D.pdf')

fig = plt.figure()
plt.plot(U, E - FE*U)
plt.scatter(ULabeled, ELabeled - FE*ULabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Energy', title=r'$f = f_e$')
bottom, top = plt.ylim()
plt.ylim(bottom=yMin, top=yMax)
label_points(ULabeled, ELabeled - FE*ULabeled)
plt.tight_layout()
ax.set_aspect(1 / ax.get_data_ratio())
plt.savefig('arch_energy_state_E.pdf')


fig = plt.figure()
plt.plot(U, E - F[-1]*U)
plt.scatter(ULabeled, ELabeled - F[-1]*ULabeled, marker='o')
ax = plt.gca()
ax.set(xlabel='Displacement', ylabel='Energy', title=r'$f = f_f$')
bottom, top = plt.ylim()
plt.ylim(bottom=yMin, top=yMax)
label_points(ULabeled, ELabeled - F[-1]*ULabeled)
plt.tight_layout()
ax.set_aspect(1 / ax.get_data_ratio())
plt.savefig('arch_energy_state_F.pdf')
