from collections import namedtuple

import jax
import jax.numpy as np

HardeningModel = namedtuple('HardeningModel', ['compute_hardening_energy_density', 'compute_flow_stress'])
RateSensitivityModel = namedtuple('RateSensitivityModel', ['compute_potential', 'compute_overstress'])

def create_hardening_model(properties):
    if 'rate sensitivity' in properties:
        S = properties['rate sensitivity stress']
        m = properties['rate sensitivity exponent']
        epsDot0 = properties['reference plastic strain rate']

        def kinetic_potential_density(eqps, eqpsOld, dt):
            return power_law_rate_sensitivity(eqps, eqpsOld, dt, S, m, epsDot0)
    else:
        kinetic_potential_density = lambda e, eo, dt: 0
    
    if properties['hardening model'] == 'linear':
        Y0 = properties['yield strength']
        H = properties['hardening modulus']

        def free_energy_density(eqps):
            return linear(eqps, Y0, H)

    elif properties['hardening model'] == 'voce':
        Y0 = properties['yield strength']
        Ysat = properties['saturation strength']
        eps0 = properties['reference plastic strain']

        def free_energy_density(eqps):
            return voce(eqps, Y0, Ysat, eps0)

    elif properties['hardening model'] == 'power law':
        Y0 = properties['yield strength']
        n = properties['hardening exponent']
        eps0 = properties['reference plastic strain']

        def free_energy_density(eqps):
            return power_law(eqps, Y0, n, eps0)

    else:
        raise ValueError('Unknown hardening model specified')
    

    def hardening(eqps, eqpsOld, dt):
        return free_energy_density(eqps) + kinetic_potential_density(eqps, eqpsOld, dt)
    

    return HardeningModel(hardening, jax.grad(hardening))


def linear(eqps, Y0, H):
    return Y0*eqps + 0.5*H*eqps**2


def voce(eqps, Y0, Ysat, eps0):
    return Ysat*eqps + (Ysat - Y0)*eps0*(np.expm1(-eqps/eps0))


def power_law(eqps, Y0, n, eps0):
    A = n*Y0*eps0/(1.0 + n)
    x = eqps/eps0
    return A*( (1.0 + x)**((n+1)/n) - 1.0 )


def power_law_rate_sensitivity(eqps, eqpsOld, dt, S, m, epsDot0):
    eqpsDot = (eqps - eqpsOld)/dt
    return m/(m + 1)*S*epsDot0*dt*(eqpsDot/epsDot0)**((m+1)/m)
