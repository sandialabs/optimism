from optimism.JaxConfig import *

from optimism.FunctionSpace import FunctionSpace
from optimism import Mechanics
from optimism.Mechanics import MechanicsFunctions

from typing import List
from typing import Optional


def setup_mechanics_functions(f_space: FunctionSpace, mat_models: dict,
                              mode2D: Optional[str] = 'plane strain') -> MechanicsFunctions:
    print('Setting up mechanics functions...')
    if len(mat_models.keys()) == 1:
        print('    Running mechanics in single block mode')
        key = list(mat_models.keys())[0]
        mech_functions = Mechanics.create_mechanics_functions(f_space, mode2D, mat_models[key])
    else:
        print('    Running mechanics in multi-block mode')
        mech_functions = Mechanics.create_multi_block_mechanics_functions(f_space, mode2D, mat_models)

    print('Finished setting up mechanics functions.\n')
    return mech_functions

def run_static_mechanics_simulation():
    pass


