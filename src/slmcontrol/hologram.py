import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union
from juliacall import Main as jl
jl.seval("using StructuredLight")


def generate_hologram(desired: ArrayLike, incoming: ArrayLike,
                      two_pi_modulation: int, x_period: Union[int, float], y_period: Union[int, float],
                      method: str = 'BesselJ1') -> NDArray[np.uint8]:
    """
    Generate a hologram used to produce the desired output.

    Args:
        desired (ArrayLike): The desired field output.
        incoming (ArrayLike): The incoming field.
        two_pi_modulation (int): The greyscale value corresponding to a 2 pi phase shift.
        x_period (Union[int, float]): The period (in pixels) of the diffraction grating in the x direction.
        y_period (Union[int, float]): The period (in pixels) of the diffraction grating in the y direction.
        method (str, optional): Hologram calculation method. 
            Possible values are:

                1. 'Simple': Method A of reference [2] 

                2. 'BesselJ1': Type 3 of reference [1] or method F of reference [2] 

                Defaults to 'BesselJ1'.

    Returns:
        NDArray[np.uint8]: The hologram.

     References:
        [1] Victor Arrizón, Ulises Ruiz, Rosibel Carrada, and Luis A. González, 
            "Pixelated phase computer holograms for the accurate encoding of scalar complex fields," 
            J. Opt. Soc. Am. A 24, 3500-3507 (2007)

        [2] Thomas W. Clark, Rachel F. Offer, Sonja Franke-Arnold, Aidan S. Arnold, and Neal Radwell, 
            "Comparison of beam generation techniques using a phase only spatial light modulator," 
            Opt. Express 24, 6249-6264 (2016)
    """
    if method == 'BesselJ1':
        _method = jl.BesselJ1
    elif method == 'Simple':
        _method = jl.Simple
    else:
        raise ValueError(
            'Invalid method. Must be either "BesselJ1" or "Simple"')
    return np.asarray(jl.generate_hologram(desired.T, incoming.T, two_pi_modulation, x_period, y_period, _method)).T
