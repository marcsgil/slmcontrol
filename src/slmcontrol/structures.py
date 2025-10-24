import numpy as np
from scipy.special import genlaguerre, hermite
from slmcontrol.typing import RealArrayLike


def lg(
    x: RealArrayLike, y: RealArrayLike, p: int = 0, L: int = 0, w: int | float = 1
) -> RealArrayLike:
    """Compute the Laguerre-Gaussian mode.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        p (int): radial index
        L (int): azymutal index
        w (int | float): waist

    Returns:
        (RealArrayLike): Laguerre-Gaussian mode.
    """
    r2 = x**2 + y**2
    phi = np.arctan2(y, x)
    return (
        np.exp(-r2 / w**2 + 1j * L * phi)
        * genlaguerre(p, abs(L))(2 * r2 / w**2)
        * np.sqrt((2 * r2 / w**2) ** abs(L))
    )


def hg(
    x: RealArrayLike, y: RealArrayLike, m: int = 0, n: int = 0, w: int | float = 1
) -> RealArrayLike:
    """Compute the Hermite-Gaussian mode.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        m (int): vertical index
        n (int): horizontal index
        w (int | float): waist

    Returns:
        (RealArrayLike): Hermite-Gaussian mode.
    """
    return (
        np.exp(-(x**2 + y**2) / w**2)
        * hermite(m)(np.sqrt(2) * x / w)
        * hermite(n)(np.sqrt(2) * y / w)
    )


def diagonal_hg(
    x: RealArrayLike, y: RealArrayLike, m: int = 0, n: int = 0, w: int | float = 1
) -> RealArrayLike:
    """Compute the diagonal Hermite-Gaussian mode.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        m (int): diagonal index
        n (int): anti-diagonal index
        w (int | float): waist

    Returns:
        (RealArrayLike): diagonal Hermite-Gaussian mode.
    """
    return (
        np.exp(-(x**2 + y**2) / w**2)
        * hermite(m)((x + y) / w)
        * hermite(n)((x - y) / w)
    )
