from optimism.JaxConfig import *
from optimism import TensorMath

MaterialModel = namedtuple('MaterialModel',
                           ['compute_energy_density',
                            'compute_output_energy_density',
                            'compute_strain_energy_density',
                            'compute_phase_potential_density',
                            'compute_initial_state',
                            'compute_state_new'])
