import numpy as np
from slmcontrol.typing import RealArrayLike, BooleanArrayLike


def rectangular_aperture(
    x: RealArrayLike, y: RealArrayLike, a: int | float, b: int | float
) -> BooleanArrayLike:
    """Rectangular aperture centered at the origin.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        a (int | float): lenght in the horizontal direction
        b (int | float): lenght in the vertical direction

    Returns:
        (BooleanArrayLike): True if the point is inside the aperture. False otherwise.
    """
    return (np.abs(x) <= a / 2) & (np.abs(y) <= b / 2)


def square(x: RealArrayLike, y: RealArrayLike, L: int | float) -> BooleanArrayLike:
    """Square apperture centered at the origin.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        L (int | float): side length

    Returns:
        (BooleanArrayLike): True if the point is inside the apperture. False otherwise.
    """
    return rectangular_aperture(x, y, L, L)


def single_slit(x: RealArrayLike, y: RealArrayLike, a: int | float) -> BooleanArrayLike:
    """Single vertical slit.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        a (int | float): slit width

    Returns:
        (BooleanArrayLike): True if the point is inside the slit. False otherwise.
    """
    return rectangular_aperture(x, y, a, np.inf)


def double_slit(
    x: np.ndarray | float | int,
    y: np.ndarray | float | int,
    a: int | float,
    d: int | float,
) -> BooleanArrayLike:
    """Double vertical slit.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        a (int | float): slit width
        d (int | float): slit separation

    Returns:
        (BooleanArrayLike): True if the point is inside the slits. False otherwise.
    """
    return single_slit(x - d / 2, y, a) | single_slit(x + d / 2, y, a)


def pupil(
    x: np.ndarray | float | int, y: np.ndarray | float | int, radius: int | float
) -> BooleanArrayLike:
    """Circular pupil centered at the origin.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        radius (int | float): radius of the pupil

    Returns:
        (BooleanArrayLike): True if the point is inside the pupil. False otherwise.
    """
    return x**2 + y**2 <= radius**2


def triangle(
    x: RealArrayLike, y: RealArrayLike, side_length: int | float
) -> BooleanArrayLike:
    """Equilateral triangular apperture centered at the origin.

    Args:
        x (RealArrayLike): x argument
        y (RealArrayLike): y argument
        side_length (int | float): side length

    Returns:
        (BooleanArrayLike): True if the point is inside the apperture. False otherwise.
    """
    sqrt3 = np.sqrt(3)
    return (y > -side_length / 2 / sqrt3) & (np.abs(x) < -y / sqrt3 + side_length / 3)
