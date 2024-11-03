import numpy as np
from juliacall import Main as jl
jl.seval("using StructuredLight")


def get_mthd_type(method):
    if method == 'BesselJ1':
        return jl.BesselJ1
    elif method == 'Simple':
        return jl.Simple
    else:
        raise ValueError(
            'Invalid method. Must be either "BesselJ1" or "Simple"')


def generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period, method='BesselJ1'):
    _method = get_mthd_type(method)
    return np.asarray(jl.generate_hologram(desired, incoming, x, y,
                                           max_modulation, x_period, y_period, _method))
