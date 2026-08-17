import numpy as np
from numpy.typing import NDArray
from scipy.special import j0, j1
from scipy.optimize import bisect
from typing import Callable, Literal


def inverse_func(f: Callable, target_value, a: float, b: float) -> float:
    """
    Finds the value x such that f(x) = target_value.

    Args:
        target_value: The desired output value of f(x).
        a (float): The lower bound for the input x.
        b (float): The upper bound for the input x.

    Returns:
        float: The value of x such that f(x) equals target_value.
    """

    # Define the function whose root we want to find
    def func_to_solve(x):
        return f(x) - target_value

    return bisect(func_to_solve, a, b)  # type: ignore


x_min_besselj1 = 0
x_max_besselj1 = 0.5818
y_min_besselj1 = 0
y_max_besselj1 = 1.82337
xs_besselj1 = np.linspace(x_min_besselj1, x_max_besselj1, 1024)
ys_besselj1 = np.empty_like(xs_besselj1)
for i, x in enumerate(xs_besselj1):
    ys_besselj1[i] = inverse_func(j1, x, y_min_besselj1, y_max_besselj1)


def inv_j1(x):
    """
    Inverse of the Bessel function of the first kind of order one, J1.

    Args:
        x (float): The value for which to compute the inverse J1.

    Returns:
        float: The value y such that J1(y) = x.
    """
    return np.interp(x, xs_besselj1, ys_besselj1)


x_min_besselj0 = 0
x_max_besselj0 = 1
y_min_besselj0 = 0
y_max_besselj0 = 2.404825557695773
xs_besselj0 = np.linspace(x_min_besselj0, x_max_besselj0, 1024)
ys_besselj0 = np.empty_like(xs_besselj0)
for i, x in enumerate(xs_besselj0):
    ys_besselj0[i] = inverse_func(j0, x, y_min_besselj0, y_max_besselj0)


def inv_j0(x):
    """
    Inverse of the Bessel function of the first kind of order zero, J0.

    The inverse is restricted to the monotonically decreasing branch between
    zero and the first positive root of J0.

    Args:
        x: The value or values for which to compute the inverse J0.

    Returns:
        The value y such that J0(y) = x.
    """
    return np.interp(x, xs_besselj0, ys_besselj0)


def generate_hologram(
    relative: NDArray,
    two_pi_modulation: int,
    x_period: int,
    y_period: int,
    method: Literal["BesselJ0", "BesselJ1"] = "BesselJ1",
) -> NDArray[np.uint8]:
    """
    Generate a hologram used to produce the desired output.

    Args:
        relative (NDArray): The relative field. This is the desired output field divided by the input field. When the input field is a plane wave, this reduces to desired output field.
        two_pi_modulation (int): The greyscale value corresponding to a 2 pi phase shift.
        x_period (int): The period (in pixels) of the diffraction grating in the x direction.
        y_period (int): The period (in pixels) of the diffraction grating in the y direction.
        method (str, optional): Hologram calculation method.
            Possible values are:
                1. 'BesselJ0': Type 2 of reference [1]
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
    relative = np.asarray(relative)
    if relative.ndim != 2:
        raise ValueError("relative must be a two-dimensional array")
    if relative.size == 0:
        raise ValueError("relative must not be empty")
    if x_period == 0 or y_period == 0:
        raise ValueError("grating periods must be non-zero")

    abs_relative = np.abs(relative)
    phase_relative = np.angle(relative)
    M = np.max(abs_relative)
    x, y = np.meshgrid(
        np.arange(relative.shape[1]), np.arange(relative.shape[0]), sparse=True
    )
    normalized_amplitude = (
        np.zeros_like(abs_relative, dtype=float) if M == 0 else abs_relative / M
    )
    carrier_phase = (
        2 * np.pi * (x / x_period + y / y_period) + phase_relative
    )

    if method == "BesselJ0":
        holo = carrier_phase + inv_j0(normalized_amplitude) * np.sin(carrier_phase)

        return np.astype(
            np.round(two_pi_modulation * np.mod(holo, 2 * np.pi) / (2 * np.pi)),
            np.uint8,
        )
    elif method == "BesselJ1":
        holo = inv_j1(x_max_besselj1 * normalized_amplitude) * np.sin(carrier_phase)

        return np.astype(
            np.round(two_pi_modulation * 0.586 * (holo / y_max_besselj1 + 1) / 2),
            np.uint8,
        )
    else:
        raise ValueError(f"Unknown hologram generation method: {method}")
