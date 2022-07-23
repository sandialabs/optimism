"""Demonstrate the material testing tools."""

from jax import numpy as np
from matplotlib import pyplot as plt

from optimism.material import MaterialUniaxialSimulator
from optimism.material import J2Plastic


E = 69e9
nu = 0.34
Y0 = 380e6
n = 5.8
ep0 = 1.5e-3
properties = {"elastic modulus": E,
              "poisson ratio": nu,
              "yield strength": Y0,
              "hardening model": "power law",
              "hardening exponent": n,
              "reference plastic strain": ep0,
              "strain measure": "logarithmic"}
material = J2Plastic.create_material_model_functions(properties)

strainRate = 1e-3

def constant_log_strain_rate(t):
    return np.expm1(strainRate*t)

maxTime = 360.0



if __name__ == "__main__":
    uniaxialData = MaterialUniaxialSimulator.run(material, constant_log_strain_rate,
                                                 maxTime, steps=100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))

    ax1.plot(uniaxialData.strainHistory[:,0,0], uniaxialData.stressHistory[:,0,0]/1e6)
    ax1.set_xlabel("Engineering strain [-]")
    ax1.set_ylabel("Nominal Stress [MPa]")
    
    ax2.plot(uniaxialData.strainHistory[:,0,0], uniaxialData.internalVariableHistory[:,J2Plastic.EQPS])
    ax2.set_xlabel("Engineering strain [-]")
    ax2.set_ylabel("Equivalent plastic strain")

    plt.savefig("aluminum.pdf")
