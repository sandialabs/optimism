import unittest

from optimism.JaxConfig import *
from optimism.material.MaterialUniaxialSimulator import MaterialUniaxialSimulator
from optimism.material import LinearElastic
from optimism.test.TestFixture import TestFixture


class MaterialUniaxialSimulatorFixture(TestFixture):

    def setUp(self):
        self.E = 100.0
        self.nu = 0.25

        properties = {"elastic modulus": self.E,
                      "poisson ratio": self.nu,
                      "strain measure": "logarithmic"}
        material = LinearElastic.create_material_model_functions(properties)
        engineering_strain_rate = 1e-3

        def strain_history(t):
            return engineering_strain_rate*t

        maxTime = 1000.0
        
        self.uniaxialTester = MaterialUniaxialSimulator(
            material, strain_history, maxTime, steps=20)


    def test_uniaxial_state_achieved(self):
        response = self.uniaxialTester.run()

        print(response.stressHistory)
        for stress in response.stressHistory[1:]:
            self.assertGreater(stress[0,0], 0.0) # axial stress is nonnegative
            self.assertTrue(np.abs(np.all(stress.ravel()[1:]) < 1e-8)) # all other components near zero


if __name__ == '__main__':
    unittest.main()
