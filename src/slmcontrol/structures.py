import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from juliacall import Main as jl
jl.seval("using StructuredLight")


def lg(x: ArrayLike, y: ArrayLike,
       p: int = 0, l: int = 0, w: Union[int, float] = 1) -> ArrayLike:
    """Compute the Laguerre-Gaussian mode.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        m (int): vertical index
        n (int): horizontal index
        w0 (Union[int, float]): waist

    Returns:
        (ArrayLike): Laguerre-Gaussian mode.
    """
    return np.asarray(jl.lg(x, y, w=w, p=p, l=l)).T


def hg(x: ArrayLike, y: ArrayLike, m: int = 0, n: int = 0, w: Union[int, float] = 1) -> ArrayLike:
    """Compute the Hermite-Gaussian mode.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        p (int): radial index
        l (int): azymutal index
        w0 (Union[int, float]): waist

    Returns:
        (ArrayLike): Hermite-Gaussian mode.
    """
    return np.asarray(jl.hg(x, y, w=w, m=m, n=n)).T


def diagonal_hg(x: ArrayLike, y: ArrayLike, m: int = 0, n: int = 0, w: Union[int, float] = 1) -> ArrayLike:
    """Compute the diagonal Hermite-Gaussian mode.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        m (int): diagonal index
        n (int): anti-diagonal index
        w0 (Union[int, float]): waist

    Returns:
        (ArrayLike): diagonal Hermite-Gaussian mode.
    """
    return np.asarray(jl.diagonal_hg(x, y, w=w, m=m, n=n)).T


def lens(x: ArrayLike, y: ArrayLike,
         fx: Union[int, float], fy: Union[int, float], k: Union[int, float] = 1) -> ArrayLike:
    """Compute the phase imposed by a lens.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        fx (Union[int, float]): focal length in the x direction
        fy (Union[int, float]): focal length in the y direction
        lamb (Union[int, float]): wavelength of incoming beam

    Returns:
        (ArrayLike): phase imposed by the lens.
    """
    return np.asarray(jl.lens(x, y, fx, fy, k=k)).T


def tilted_lens(x: ArrayLike, y: ArrayLike,
                f: Union[int, float], ϕ: Union[int, float], k: Union[int, float] = 1) -> ArrayLike:
    """Compute the phase imposed by a tilted spherical lens.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        f (Union[int, float]): focal length
        theta (Union[int, float]): tilting angle
        lamb (Union[int, float]): wavelength of incoming beam

    Returns:
        (ArrayLike): phase imposed by the tilted spherical lens
    """

    return np.asarray(jl.tilted_lens(x, y, f, ϕ, k=k)).T


def rectangular_apperture(x: ArrayLike, y: ArrayLike, a: Union[int, float], b: Union[int, float]) -> ArrayLike:
    """Rectangular apperture centered at the origin.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        a (Union[int, float]): lenght in the horizontal direction
        b (Union[int, float]): lenght in the vertical direction

    Returns:
        (ArrayLike): True if the point is inside the apperture. False otherwise.
    """
    return np.asarray(jl.rectangular_apperture(x, y, a, b)).T


def square(x: ArrayLike, y: ArrayLike, l: Union[int, float]) -> ArrayLike:
    """Square apperture centered at the origin.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        l (Union[int, float]): side length

    Returns:
        (ArrayLike): True if the point is inside the apperture. False otherwise.
    """
    return np.asarray(jl.square(x, y, l)).T


def single_slit(x: ArrayLike, y: ArrayLike, a: Union[int, float]) -> ArrayLike:
    """Single vertical slit.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        a (Union[int, float]): slit widht

    Returns:
        (ArrayLike): True if the point is inside the slit. False otherwise.
    """
    return np.asarray(jl.single_slit(x, y, a)).T


def double_slit(x: ArrayLike, y: ArrayLike, a: Union[int, float], d: Union[int, float]) -> ArrayLike:
    """Double vertical slit.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        a (Union[int, float]): slit widht
        d (Union[int, float]): slit separation

    Returns:
        (ArrayLike): True if the point is inside the slits. False otherwise.
    """
    return np.asarray(jl.double_slit(x, y, a, d)).T


def pupil(x: ArrayLike, y: ArrayLike, radius: Union[int, float]) -> ArrayLike:
    """Circular pupil centered at the origin.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        radius (Union[int, float]): radius of the pupil

    Returns:
        (ArrayLike): True if the point is inside the pupil. False otherwise.
    """
    return np.asarray(jl.pupil(x, y, radius)).T


def triangle(x: ArrayLike, y: ArrayLike, side_length: Union[int, float]) -> ArrayLike:
    """Equilateral triangular apperture centered at the origin.

    Args:
        x (ArrayLike): x argument
        y (ArrayLike): y argument
        side_length (Union[int, float]): side length

    Returns:
        (ArrayLike): True if the point is inside the apperture. False otherwise.
    """
    return np.asarray(jl.triangle(x, y, side_length)).T
