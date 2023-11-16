from multimethod import multimethod
import numpy as np
from slmcontrol import *


@multimethod
def linear_combination(coefficients, basis_name: str, config_path: str, w0: float):
    """Calculate linear combination of order 'len(coefficients)-1' on the specified basis

    Args:
        coefficients (list): list of coefficients
        basis_name (str): name of the basis. Possible values are 'lg', 'hg' and 'diagonal_hg'
        config_path (str): Path for the configuration file of the SLM
        w0 (float): waist

    Raises:
        ValueError: 'basis_name' is not one of the specified values

    Returns:
        (array_like): linear combination
    """
    x, y = build_grid(config_path)
    return linear_combination(coefficients, basis_name, x, y, w0)


@multimethod
def linear_combination(coefficients, basis_name: str, x, y, w0: float):
    """Calculate linear combination of order `len(coefficients)-1` on the specified basis

    Args:
        coefficients (list): list of coefficients
        basis_name (str): name of the basis. Possible values are 'lg', 'hg' and 'diagonal_hg'
        x (array_like): x grid
        y (array_like): y grid
        w0 (float): waist

    Raises:
        ValueError: 'basis_name' is not one of the specified values

    Returns:
        (array_like): linear combination
    """
    if basis_name not in ('lg', 'hg', 'diagonal_hg'):
        raise ValueError(
            "Known 'basis_name' are 'lg', 'hg', 'diagonal_hg'. Got %s." % basis_name
        )

    order = len(coefficients) - 1

    if basis_name == 'lg':
        basis = [lg(x, y, int(np.minimum(k, order-k)), 2*k - order, w0)
                 for k in range(order+1)]
    elif basis_name == 'hg':
        basis = [hg(x, y, order-k, k, w0) for k in range(order+1)]
    elif basis_name == 'diagonal_hg':
        basis = [diagonal_hg(x, y, order-k, k, w0) for k in range(order+1)]

    return linear_combination(coefficients, basis)
