import numpy as np
from juliacall import Main as jl
jl.seval("using StructuredLight")


def lg(x, y, w=1, p=0, l=0):
    """Compute the Hermite-Gaussian mode.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        m (int): vertical index
        n (int): horizontal index
        w0 (Real): waist

    Returns:
        (array_like): Hermite-Gaussian mode.
    """
    return np.asarray(jl.lg(x, y, w=w, p=p, l=l))


def hg(x, y, w=1, m=0, n=0):
    """Compute the Laguerre-Gaussian mode.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        p (int): radial index
        l (int): azymutal index
        w0 (Real): waist

    Returns:
        (array_like): Laguerre-Gaussian mode.
    """
    return np.asarray(jl.hg(x, y, w=w, m=m, n=n))


def diagonal_hg(x, y, w, m, n):
    """Compute the diagonal Hermite-Gaussian mode.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        m (int): diagonal index
        n (int): anti-diagonal index
        w0 (Real): waist

    Returns:
        (array_like): diagonal Hermite-Gaussian mode.
    """
    return np.asarray(jl.diagonal_hg(x, y, w=w, m=m, n=n))


def lens(x, y, fx, fy, k=1):
    """Compute the phase imposed by a lens.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        fx (Real): focal length in the x direction
        fy (Real): focal length in the y direction
        lamb (Real): wavelength of incoming beam

    Returns:
        (array_like): phase imposed by the lens.
    """
    return np.asarray(jl.lens(x, y, fx, fy, k=k))


def tilted_lens(x, y, f, ϕ, k=1):
    """Compute the phase imposed by a tilted spherical lens.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        f (Real): focal length
        theta (Real): tilting angle
        lamb (Real): wavelength of incoming beam

    Returns:
        (array_like): phase imposed by the tilted spherical lens
    """

    return np.asarray(jl.tilted_lens(x, y, f, ϕ, k=k))


def rectangular_apperture(x, y, a, b):
    """Rectangular apperture centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        a (Real): lenght in the horizontal direction
        b (Real): lenght in the vertical direction

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    return np.asarray(jl.rectangular_apperture(x, y, a, b))


def square(x, y, l):
    """Square apperture centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        l (Real): side length

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    return np.asarray(jl.square(x, y, l))


def single_slit(x, y, a):
    """Single vertical slit.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        a (Real): slit widht

    Returns:
        (array_like): True if the point is inside the slit. False otherwise.
    """
    return np.asarray(jl.single_slit(x, y, a))


def double_slit(x, y, a, d):
    """Double vertical slit.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        a (Real): slit widht
        d (Real): slit separation

    Returns:
        (array_like): True if the point is inside the slits. False otherwise.
    """
    return np.asarray(jl.double_slit(x, y, a, d))


def pupil(x, y, radius):
    """Circular pupil centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        radius (Real): radius of the pupil

    Returns:
        (array_like): True if the point is inside the pupil. False otherwise.
    """
    return np.asarray(jl.pupil(x, y, radius))


def triangle(x, y, side_length):
    """Equilateral triangular apperture centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        side_length (Real): side length

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    return np.asarray(jl.triangle(x, y, side_length))
