import numpy as np
from scipy import special
from scipy.special import factorial
from multimethod import multimethod
from slmcontrol.hologram import build_grid


@multimethod
def hg(config_path: str, m: int, n: int, w0):
    """Compute the Hermite-Gaussian mode.

    Args:
        config_path (str): Path for the configuration file of the SLM
        m (int): vertical index
        n (int): horizontal index
        w0 (Real): waist

    Returns:
        (array_like): Hermite-Gaussian mode.
    """

    x, y = build_grid(config_path)
    return hg(x, y, m, n, w0)


@multimethod
def hg(x, y, m: int, n: int, w0):
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

    pm = special.hermite(m)
    pn = special.hermite(n)

    N = np.sqrt(2 / (np.pi * 2**(m+n) * factorial(m) * factorial(n))) / w0

    return N * pm(np.sqrt(2) * x / w0) * pn(np.sqrt(2) * y / w0) * np.exp(-(x**2+y**2)/w0**2)


@multimethod
def lg(config_path: str, p: int, l: int, w0):
    """Compute the Laguerre-Gaussian mode.

    Args:
        config_path (str): Path for the configuration file of the SLM
        p (int): radial index
        l (int): azymutal index
        w0 (Real): waist

    Returns:
        (array_like): Laguerre-Gaussian mode.
    """
    x, y = build_grid(config_path)
    return lg(x, y, p, l, w0)


@multimethod
def lg(x, y, p: int, l: int, w0):
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
    lag = special.genlaguerre(p, abs(l))

    N = np.sqrt(2 * factorial(p) / np.pi / factorial(p + np.abs(l))) / w0

    r = np.sqrt((x**2 + y**2)) / w0
    return (
        N * (np.sqrt(2)*r)**(abs(l))
        * np.exp(-r**2) * lag(2*r**2)
        * np.exp(1j*l*np.arctan2(y, x)))


@multimethod
def diagonal_hg(config_path: str, m: int, n: int, w0):
    """Compute the diagonal Hermite-Gaussian mode.

    Args:
        config_path (str): Path for the configuration file of the SLM
        m (int): diagonal index
        n (int): anti-diagonal index
        w0 (Real): waist

    Returns:
        (array_like): diagonal Hermite-Gaussian mode.
    """
    x, y = build_grid(config_path)
    return diagonal_hg(x, y, m, n, w0)


@multimethod
def diagonal_hg(x, y, m: int, n: int, w0):
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
    return hg((x-y)/np.sqrt(2), (x+y)/np.sqrt(2), m, n, w0)


@multimethod
def lens(config_path: str, fx, fy, lamb):
    """_summary_

    Args:
        config_path (str): Path for the configuration file of the SLM
        fx (Real): focal length in the x direction
        fy (Real): focal length in the y direction
        lamb (Real): wavelength of incoming beam

    Returns:
        (array_like): phase imposed by the lens.
    """
    x, y = build_grid(config_path)
    return lens(x, y, fx, fy, lamb)


@multimethod
def lens(x, y, fx, fy, lamb):
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
    return np.exp(-1j*np.pi/lamb*((x**2)/fx + (y**2)/fy))


@multimethod
def tilted_lens(config_path: str, f, theta, lamb):
    """Compute the phase imposed by a tilted spherical lens.

    Args:
        config_path (str): Path for the configuration file of the SLM
        f (Real): focal length
        theta (Real): tilting angle
        lamb (Real): wavelength of incoming beam

    Returns:
        (array_like): phase imposed by the tilted spherical lens
    """
    fx = f*np.cos(theta)
    fy = f/np.cos(theta)
    return tilted_lens(config_path, fx, fy, lamb)


@multimethod
def tilted_lens(x, y, f, theta, lamb):
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
    fx = f*np.cos(theta)
    fy = f/np.cos(theta)
    return lens(x, y, fx, fy, lamb)


@multimethod
def rectangular_apperture(config_path: str, a, b):
    """Rectangular apperture centered at the origin.

    Args:
        config_path (str): Path for the configuration file of the SLM
        a (Real): lenght in the horizontal direction
        b (Real): lenght in the vertical direction

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    x, y = build_grid(config_path)
    rectangular_apperture(x, y, a, b)


@multimethod
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


@multimethod
def square(config_path: str, l):
    """Square apperture centered at the origin.

    Args:
        config_path (str): Path for the configuration file of the SLM
        l (Real): side length

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    x, y = build_grid(config_path)
    return square(x, y, l)


@multimethod
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


@multimethod
def single_slit(config_path: str, a):
    """Single vertical slit.

    Args:
        config_path (str): Path for the configuration file of the SLM
        a (Real): slit widht

    Returns:
        (array_like): True if the point is inside the slit. False otherwise.
    """
    x, y = build_grid(config_path)
    return single_slit(x, y, a)


@multimethod
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


@multimethod
def double_slit(config_path: str, a, d):
    """Double vertical slit.

    Args:
        config_path (str): Path for the configuration file of the SLM
        a (Real): slit widht
        d (Real): slit separation

    Returns:
        (array_like): True if the point is inside the slits. False otherwise.
    """
    x, y = build_grid(config_path)
    return double_slit(x, y, a, d)


@multimethod
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


@multimethod
def pupil(config_path: str, radius):
    """Circular pupil centered at the origin.

    Args:
        config_path (str): Path for the configuration file of the SLM
        radius (Real): radius of the pupil

    Returns:
        (array_like): True if the point is inside the pupil. False otherwise.
    """
    x, y = build_grid(config_path)
    return pupil(x, y, radius)


@multimethod
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


@multimethod
def triangle(config_path: str, side_length):
    """Equilateral triangular apperture centered at the origin.

    Args:
        config_path (str): Path for the configuration file of the SLM
        side_length (Real): side length

    Returns:
        (array_like): True if the point is inside the apperture. False otherwise.
    """
    x, y = build_grid(config_path)
    return triangle(x, y, side_length)


@multimethod
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
