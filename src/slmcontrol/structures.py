import numpy as np
from scipy.special import genlaguerre, hermite, factorial
from numpy.typing import NDArray


def lg(x: NDArray, y: NDArray, p: int = 0, l: int = 0, w: int | float = 1) -> NDArray:  # noqa: E741
    """Compute the Laguerre-Gaussian mode.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        p (int): radial index
        l (int): azymutal index
        w (int | float): waist

    Returns:
        (NDArray): Laguerre-Gaussian mode.
    """
    normalization = np.sqrt(2 * factorial(p) / np.pi / factorial(p + np.abs(l))) / w
    r2 = x**2 + y**2
    phi = np.arctan2(y, x)
    return (
        normalization
        * np.exp(-r2 / w**2 + 1j * l * phi)
        * genlaguerre(p, abs(l))(2 * r2 / w**2)
        * np.sqrt((2 * r2 / w**2) ** abs(l))
    )


def hg(x: NDArray, y: NDArray, m: int = 0, n: int = 0, w: int | float = 1) -> NDArray:
    """Compute the Hermite-Gaussian mode.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        m (int): vertical index
        n (int): horizontal index
        w (int | float): waist

    Returns:
        (NDArray): Hermite-Gaussian mode.
    """
    normalization = np.sqrt(2 / np.pi / 2**(m+n) / factorial(m) / factorial(n)) / w

    return (
        normalization *
        np.exp(-(x**2 + y**2) / w**2)
        * hermite(m)(np.sqrt(2) * x / w)
        * hermite(n)(np.sqrt(2) * y / w)
    )


def diagonal_hg(
    x: NDArray, y: NDArray, m: int = 0, n: int = 0, w: int | float = 1
) -> NDArray:
    """Compute the diagonal Hermite-Gaussian mode.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        m (int): diagonal index
        n (int): anti-diagonal index
        w (int | float): waist

    Returns:
        (NDArray): diagonal Hermite-Gaussian mode.
    """
    normalization = np.sqrt(2 / np.pi / 2**(m+n) / factorial(m) / factorial(n)) / w
    return (
        normalization
        * np.exp(-(x**2 + y**2) / w**2)
        * hermite(m)((x + y) / w)
        * hermite(n)((x - y) / w)
    )


def lens(
    x: NDArray,
    y: NDArray,
    fx: int | float,
    fy: int | float,
    k: int | float = 1,
) -> NDArray:
    """Compute the phase imposed by a lens.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        fx (int | float): focal length in the x direction
        fy (int | float): focal length in the y direction
        k (int | float): wavenumber of incoming beam

    Returns:
        (NDArray): phase imposed by the lens.
    """
    return np.exp(-1j * k * (x**2 / (2 * fx) + y**2 / (2 * fy)))
