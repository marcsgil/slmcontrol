from multimethod import multimethod
import numpy as np


@multimethod
def linear_combination(coefficients, basis):
    """Calculate a linear combinantion

    Args:
        coefficients (list): list containing the coefficients
        basis (list): list containing the basis elements

    Returns:
        (array_like): array defined by the linear combination
    """
    return np.sum([c * b for (c, b) in zip(coefficients, basis)], axis=0)
