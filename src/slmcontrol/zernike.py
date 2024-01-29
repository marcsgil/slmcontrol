import numpy as np
from scipy.special import binom
import configparser
from slmcontrol.hologram import build_grid
from functools import singledispatch


def R(r, n, m):
    """
    Compute the radial polynomial for Zernike polynomials.

    Args:
        r (array_like): Radial coordinate.
        n (int): Radial order of the Zernike polynomial.
        m (int): Azimuthal order of the Zernike polynomial.

    Returns:
        (array_like): The value of the radial polynomial.
    """
    s = 0.
    if (n-m) % 2 == 0:
        for k in range((n-m) // 2 + 1):
            s += (-1)**k * binom(n-k, k) * \
                binom(n-2*k, (n-m) // 2 - k) * r**(n-2*k)
    return s


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
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)
    assert m in range(-n, n+1,
                      2), "The index combination is does not define a valid polynomial"
    if m >= 0:
        return np.sqrt(2*(n+1))*R(r, n, m)*np.cos(m*phi)
    else:
        return np.sqrt(2*(n+1))*R(r, n, -m)*np.sin(m*phi)


@singledispatch
def zernike_combination(x, y, indices, coefficients):
    """
    Compute a linear combination of Zernike polynomials.

    Args:
        x (array_like): x-coordinates.
        y (array_like): y-coordinates.
        indices (list of tuples): Each tuple contains the radial and azimuthal orders of a Zernike polynomial.
        coefficients (list of floats): Coefficients for the linear combination.

    Returns:
        (array_like): The values of the linear combination of Zernike polynomials at the points (x, y).

    Note:
        There is also a method `zernike_combination(config_path: str)` where `config_path` is the path for the configuration file.
    """
    p = np.zeros((y.shape[0], x.shape[1]))
    for (c, idx) in zip(coefficients, indices):
        p += c*zernike(x, y, idx[0], idx[1])
    return p


@zernike_combination.register
def _(config_path: str):
    config = configparser.ConfigParser()
    config.read(config_path)

    x, y = build_grid(config_path)

    indices = [[1, -1], [1, 1],    [2, -2], [2, 0],
               [2, 2],    [3, -3], [3, -1], [3, 1], [3, 3]]
    coefficients = [config['zernike'].getfloat(
        key) for key in config['zernike']]

    return zernike_combination(x, y, indices, coefficients)
