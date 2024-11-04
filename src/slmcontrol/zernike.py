import numpy as np
from juliacall import Main as jl
jl.seval("using StructuredLight")


def zernike(x, y, n, m):
    """
    Compute the Zernike polynomial of order (n, m) at the points (x, y).

    Args:
        x (array_like): x-coordinates.
        y (array_like): y-coordinates.
        n (int): Radial order of the Zernike polynomial.
        m (int): Azimuthal order of the Zernike polynomial.

    Returns:
        (array_like): The values of the Zernike polynomial at the points (x, y).
    """
    return np.asarray(jl.zernike_polynomial(x, y, n, m)).T
