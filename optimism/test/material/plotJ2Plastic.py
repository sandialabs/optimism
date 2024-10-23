from matplotlib import pyplot as plt

from optimism.JaxConfig import *
from optimism.material.MaterialPointUniaxialSimulator import MaterialPointUniaxialSimulator
from optimism.material import J2Plastic as J2


def plot_J2_uniaxial():
    E = 200.0e3
    nu = 0.25
    Y0 = 350.0
    H = 1e-1*E
    Ysat = 800.0
    eps0 = 0.01
    props = {'elastic modulus': E,
             'poisson ratio': nu,
             'yield strength': Y0,
             'hardening model': 'voce',
             'hardening modulus': H,
             'saturation strength': Ysat,
             'reference plastic strain': eps0}
    materialModel = J2.create_material_model_functions(props)
    maxStrain = 20.0*Y0/E
    strainRate = 1.0
    simulator = MaterialPointUniaxialSimulator(materialModel, maxStrain, strainRate, steps=20)
    out = simulator.run()
    plt.plot(out.strainHistory, out.stressHistory, marker='o')
    plt.show()

if __name__ == "__main__":
    plot_J2_uniaxial()
