import numpy as np
from scipy import special
from scipy.special import factorial, binom
from functools import singledispatch


def hg(x, y, m, n, w0):
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


def lg(x, y, p, l, w0):
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
        * np.exp(1j*l*np.arctan2(y, x))
    )


def diagonal_hg(x, y, m, n, w0):
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
    return hg((x+y)/np.sqrt(2), (x-y)/np.sqrt(2), m, n, w0)


def b(m, n, k):
    """
    Calculate the value the coefficients b(m, n, k) that converts between Hermite-Gaussian and diagonal Hermite-Gaussian modes.

    Parameters:
        m (int): The value of m.
        n (int): The value of n.
        k (int): The value of k.

    Returns:
        (float): The calculated value of b(m, n, k).

    Reference:
        1. Beijersbergen, M. W., Allen, L., van der Veen, H. E. L. O. & Woerdman, J. P. Astigmatic laser mode converters and transfer of orbital angular momentum. Optics Communications 96, 123-132 (1993).
    """

    N = m + n
    prefactor = np.sqrt(factorial(k) * factorial(N-k) /
                        (2**N * factorial(m) * factorial(n)))

    max = np.minimum(n, k)
    min = np.maximum(0, k-m)

    return prefactor * np.sum([(-1)**j * binom(n, j) * binom(m, k-j) for j in range(min, max+1)])


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


@singledispatch
def linear_combination(elements, coefficients):
    """Calculate a linear combinantion

    Args:
        coefficients (list): list containing the coefficients of the linear combination
        elements (list): list containing the elements of the linear combination

    Returns:
        (array_like): array defined by the linear combination

    Note:
        There is also a version of this function with signature `linear_combination(basis_name: str, coefficients, x, y, w0)`.
        It is a wrapper for this function that uses as elements the basis of fixed order modes calculated by `fixed_order_basis`.
    """
    return np.sum([c * b for (c, b) in zip(coefficients, elements)], axis=0)


def fixed_order_basis(basis_name, order, x, y, w0):
    """Calculate the basis of fixed order.

    Args:
        basis_name (str): Name of the chosen basis. Can be 'lg', 'hg' or 'diagonal_hg'.
        order (int): Order of the modes.
        x (array_like): x argument
        y (array_like): y argument
        w0 (float): Waist of the modes.

    Raises:
        ValueError: If the basis_name is not 'lg', 'hg' or 'diagonal_hg'.

    Returns:
        (array_like): A list containing the elements of the basis.
    """
    if basis_name not in ('lg', 'hg', 'diagonal_hg'):
        raise ValueError(
            "Known 'basis_name' are 'lg', 'hg', 'diagonal_hg'. Got %s." % basis_name
        )
    if basis_name == 'lg':
        return [lg(x, y, int(np.minimum(k, order-k)), 2*k - order, w0)
                for k in range(order+1)]
    elif basis_name == 'hg':
        return [hg(x, y, order-k, k, w0) for k in range(order+1)]
    elif basis_name == 'diagonal_hg':
        return [diagonal_hg(x, y, order-k, k, w0) for k in range(order+1)]


@linear_combination.register
def _(basis_name: str, coefficients, x, y, w0):
    order = len(coefficients) - 1
    basis = fixed_order_basis(basis_name, order, x, y, w0)
    return linear_combination(basis, coefficients)
