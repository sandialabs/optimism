from optimism.JaxConfig import *
from optimism import TensorMath

MatProps = namedtuple('MatProps', ['props','num_props','num_states'])

MaterialModel = namedtuple('MaterialModel',
                           ['compute_energy_density', 'compute_output_energy_density',
                            'compute_initial_state', 'compute_state_new'])
