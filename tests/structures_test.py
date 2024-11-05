import pytest
import numpy as np
from slmcontrol.structures import (
    lg, hg, diagonal_hg, lens, tilted_lens, rectangular_apperture,
    square, single_slit, double_slit, pupil, triangle
)


@pytest.mark.parametrize("function,args", [
    (lg, []),
    (hg, []),
    (diagonal_hg, [1, 1, 1]),
    (lens, [1.0, 1.0]),
    (tilted_lens, [1.0, 1.0]),
    (rectangular_apperture, [1.0, 1.0]),
    (square, [1.0]),
    (single_slit, [1.0]),
    (double_slit, [1.0, 1.0]),
    (pupil, [1.0]),
    (triangle, [1.0]),
])
def test_structure_functions(xy_grid, test_shape, function, args):
    """Test various structure generation functions."""
    x, y = xy_grid
    result = function(x, y, *args)
    assert isinstance(result, np.ndarray)
    assert result.shape == test_shape
