from collections import namedtuple

MatProps = namedtuple('MatProps', ['props','num_props','num_states'])

MaterialModel = namedtuple('MaterialModel',
                           ['compute_energy_density', 'compute_initial_state', 'compute_state_new',
                            'density','compute_material_qoi'],
                           defaults=(0.0,))
