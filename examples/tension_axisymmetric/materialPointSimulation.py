from matplotlib import pyplot as plt

from optimism.JaxConfig import *
from optimism import TensorMath
from optimism.phasefield import SandiaModel as MaterialModel
from optimism.phasefield.MaterialPointSimulator import MaterialPointSimulator

from properties import props

maxStrain = 0.1
strainRate = 1.e-3

# take care to resolve the elastic region
yieldStrain = props['yield strength']/props['elastic modulus']
steps = int(maxStrain // (0.5*yieldStrain))

material = MaterialModel.create_material_model_functions(props)
simulator = MaterialPointSimulator(material, maxStrain, strainRate, steps=steps)

output = simulator.run()


plt.plot(output.strainHistory, output.stressHistory)
plt.xlabel('strain')
plt.ylabel('stress')
plt.tight_layout()
plt.show()
