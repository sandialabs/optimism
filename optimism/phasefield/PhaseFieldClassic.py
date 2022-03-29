# TODO: update this model to current conventions (or remove)
# BT 08/01/2021
# We haven't used this model in a while. We can update it, or perhaps just drop
# it. I am eliminating all tests for now.

from sys import float_info

from optimism.JaxConfig import *
from optimism.TensorMath import *
from optimism.Mesh import compute_element_field_gradient

# index enumerations
PHASE=0
E22=1
E33=2
NUMUNKNOWNS=3

DOFS=slice(0,NUMUNKNOWNS)
NUMSTATE=NUMUNKNOWNS

def degradation(phase):
    return (1.-phase)*(1.-phase)


def intact_strain_energy_density(props, strain):
    dil = np.trace(strain)
    dev = compute_deviatoric_tensor(strain).ravel()
    return 0.5*props['kappa'] * dil**2 + props['mu'] * np.dot(dev,dev)


def strain_energy_density(props, strain, phase):
    energy = degradation(phase) * intact_strain_energy_density(props, strain)
    return energy


def phase_potential_density(props, phase, gradPhase):
    gradPhaseNormSquared = np.dot(gradPhase, gradPhase)
    return 0.5*props['Gc']/props['L'] * ( phase**2 + pow(props['L'],2) * gradPhaseNormSquared )


def free_energy_density(props, strain, phase, gradPhase):
    return strain_energy_density(props, strain, phase) + phase_potential_density(props, phase, gradPhase)


def compute_element_energy(compute_free_energy_density, U, shapeGrad, conn):
    nodalDisp = U[conn,:2]
    dispGrad = np.tensordot(nodalDisp, shapeGrad, axes=[0,0]) 
    strain = np.zeros((3,3)).at[0:2,0:2].set(dispGrad)
    nodalPhase = U[conn,2]
    phaseGrad = np.tensordot(nodalPhase, shapeGrad, axes=[0,0]) 
    phase = np.average(nodalPhase)
    return compute_free_energy_density(strain, phase, phaseGrad)

def compute_total_energy(compute_free_energy_density, U, mesh):
    energyDensities = vmap(compute_element_energy, (None,None,0,0))(compute_free_energy_density, U, mesh.shapeGrads, mesh.conns)
    return np.dot(energyDensities, mesh.vols)
