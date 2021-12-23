from optimism.JaxConfig import *
from optimism import Math


Params = namedtuple('FrictionParams', ['mu', 'sReg'])


def compute_friction_energy_from_perp_slip(sPerp, frictionParams):
    sReg = frictionParams.sReg
    sPerpSquared = sPerp@sPerp

    fEnergy = np.where( sPerpSquared <= sReg*sReg,
                        sPerpSquared / (2*sReg),
                        Math.safe_sqrt(sPerpSquared) - 0.5*sReg )
    
    return frictionParams.mu * fEnergy
