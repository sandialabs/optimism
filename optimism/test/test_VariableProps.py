
from optimism import ReadExodusMesh
from . import MeshFixture
import pathlib

# is MeshFixture the proper one? is one necessary? 
class TestVariableProps(MeshFixture.MeshFixture):
    def setUp(self):
        # assign input testing mesh 
        self.input_mesh = pathlib.Path(__file__).parent.joinpath('read_variable_material_property_test_Seeded.exo')


    # test whether the values pulled from the mesh match the expected ones
    def test_read_variable_props(self):
        # assign the properties of each element to an array 
        varProps = ReadExodusMesh.read_exodus_mesh_element_properties(self.input_mesh, ['light_dose'], blockNum=1)
        
        # define the expected values of each element
        assignedValues = [500.0, 20.0, 260.0, 260.0]
        
        # test whether each element has the property we expect 
        for i in range(len(assignedValues)):
            self.assertEqual(varProps[i], assignedValues[i])




if __name__ == "__main__":
    unittest.main()
