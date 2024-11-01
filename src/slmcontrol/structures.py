import numpy as np
from juliacall import Main as jl
jl.seval("using StructuredLight")


def lg(x, y, w=1, p=0, l=0):
    return jl.lg(x, y, w=w, p=p, l=l)


def hg(x, y, w=1, m=0, n=0):
    return jl.hg(x, y, w=w, m=m, n=n)


def diagonal_hg(x, y, w, m, n):
    return jl.diagonal_hg(x, y, w=w, m=m, n=n)


def lens(x, y, fx, fy, k=1):
    return jl.lens(x, y, fx, fy, k=k)


def tilted_lens(x, y, f, ϕ, k=1):
    return jl.tilted_lens(x, y, f, ϕ, k=k)


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
    return np.vectorize(lambda x, y: np.abs(x) <= a/2 and np.abs(y) <= b/2)(x, y)


def square(x, y, l):
    """Square apperture centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        l (Real): side length

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    return rectangular_apperture(x, y, l, l)


def single_slit(x, y, a):
    """Single vertical slit.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        a (Real): slit widht

    Returns:
        (array_like): True if the point is inside the slit. False otherwise.
    """
    return rectangular_apperture(x, y, a, np.inf)


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
    return rectangular_apperture(x - d/2, y, a, np.inf) + rectangular_apperture(x + d/2, y, a, np.inf)


def pupil(x, y, radius):
    """Circular pupil centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        radius (Real): radius of the pupil

    Returns:
        (array_like): True if the point is inside the pupil. False otherwise.
    """
    return np.vectorize(lambda x, y: x**2+y**2 <= radius**2)(x, y)


def triangle(x, y, side_length):
    """Equilateral triangular apperture centered at the origin.

    Args:
        x (array_like): x argument
        y (array_like): y argument
        side_length (Real): side length

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    def is_inside(x, y):
        return y > -side_length/2/np.sqrt(3) and np.abs(x) < -y/np.sqrt(3) + side_length / 3
    return np.vectorize(is_inside)(x, y)
