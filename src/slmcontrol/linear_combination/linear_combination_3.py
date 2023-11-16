from multimethod import multimethod
import numpy as np
from slmcontrol import *


@multimethod
def linear_combination(coefficients, indices, basis_name: str, config_path: str, w0: float):
    """Generate a linear combination using the coefficients and the list of indices, which refer to the specified basis

    Args:
        coefficients (list): list of coefficients
        indices (list): list of indices
        basis_name (str): name of the basis. Possible values are 'lg', 'hg' and 'diagonal_hg'
        config_path (str): Path for the configuration file of the SLM
        w0 (float): waist

    Raises:
        ValueError: 'basis_name' is not one of the specified values

    Returns:
        (array_like): Linear combination
    """
    x, y = build_grid(config_path)
    return linear_combination(coefficients, indices, basis_name, x, y, w0)


@multimethod
def linear_combination(coefficients: list, indices: list, basis_name: str, x, y, w0: float):
    """Generate a linear combination using the coefficients and the list of indices, which refer to the specified basis

    Args:
        coefficients (list): list of coefficients
        indices (list): list of indices
        basis_name (str): name of the basis. Possible values are 'lg', 'hg' and 'diagonal_hg'
        x (array_like): x grid
        y (array_like): y grid
        w0 (float): waist

    Raises:
        ValueError: 'basis_name' is not one of the specified values

    Returns:
        (array_like): Linear combination
    """
    if basis_name not in ('lg', 'hg', 'diagonal_hg'):
        raise ValueError(
            "Known 'basis_name' are 'lg', 'hg', 'diagonal_hg'. Got %s." % basis_name
        )

    def basis(i1, i2, w0):
        if basis_name == 'lg':
            return lg(x, y, i1, i2, w0)
        elif basis_name == 'hg':
            return hg(x, y, i1, i2, w0)
        elif basis_name == 'diagonal_hg':
            return diagonal_hg(x, y, i1, i2, w0)

    return np.sum([c * basis(i[0], i[1], w0) for (c, i) in zip(coefficients, indices)], axis=0)
