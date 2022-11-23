from jax import numpy as np
from matplotlib import pyplot as plt
from optimism.material import MaterialUniaxialSimulator
from optimism.material import Gent


properties = {
    'shear modulus': 1.0e6,
    'bulk modulus': 1.0e9,
    'Jm parameter': 3.0
}

material = Gent.create_material_functions(properties)

strainRate = 1e-3

def constant_log_strain_rate(t):
    return np.expm1(strainRate * t)


maxTime = 720.0


if __name__ == '__main__':
    uniaxialData = MaterialUniaxialSimulator.run(material, constant_log_strain_rate, maxTime, steps=100)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(uniaxialData.strainHistory[:,0,0], uniaxialData.stressHistory[:,0,0]/1e6)
    ax1.set_xlabel("Engineering strain [-]")
    ax1.set_ylabel("Nominal Stress [MPa]")

    plt.savefig('dummy_material.pdf')