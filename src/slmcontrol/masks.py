import numpy as np
from numpy.typing import NDArray


def rectangular_aperture(
    x: NDArray, y: NDArray, a: int | float, b: int | float
) -> NDArray[np.bool]:
    """Rectangular aperture centered at the origin.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        a (int | float): lenght in the horizontal direction
        b (int | float): lenght in the vertical direction

    Returns:
        (NDArray): True if the point is inside the aperture. False otherwise.
    """
    return (np.abs(x) <= a / 2) & (np.abs(y) <= b / 2)


def square(x: NDArray, y: NDArray, L: int | float) -> NDArray[np.bool]:
    """Square apperture centered at the origin.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        L (int | float): side length

    Returns:
        (NDArray): True if the point is inside the apperture. False otherwise.
    """
    return rectangular_aperture(x, y, L, L)


def single_slit(x: NDArray, y: NDArray, a: int | float) -> NDArray[np.bool]:
    """Single vertical slit.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        a (int | float): slit width

    Returns:
        (NDArray): True if the point is inside the slit. False otherwise.
    """
    return rectangular_aperture(x, y, a, np.inf)


def double_slit(
    x: NDArray,
    y: NDArray,
    a: int | float,
    d: int | float,
) -> NDArray[np.bool]:
    """Double vertical slit.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        a (int | float): slit width
        d (int | float): slit separation

    Returns:
        (NDArray): True if the point is inside the slits. False otherwise.
    """
    return single_slit(x - d / 2, y, a) | single_slit(x + d / 2, y, a)


def pupil(x: NDArray, y: NDArray, radius: int | float) -> NDArray[np.bool]:
    """Circular pupil centered at the origin.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        radius (int | float): radius of the pupil

    Returns:
        (NDArray): True if the point is inside the pupil. False otherwise.
    """
    return x**2 + y**2 <= radius**2


def triangle(x: NDArray, y: NDArray, side_length: int | float) -> NDArray[np.bool]:
    """Equilateral triangular apperture centered at the origin.

    Args:
        x (NDArray): x argument
        y (NDArray): y argument
        side_length (int | float): side length

    Returns:
        (NDArray): True if the point is inside the apperture. False otherwise.
    """
    sqrt3 = np.sqrt(3)
    return (y > -side_length / 2 / sqrt3) & (np.abs(x) < -y / sqrt3 + side_length / 3)
