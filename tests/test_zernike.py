import pytest
from slmcontrol.zernike import zernike
import numpy as np


def test_zernike(xy_grid):
    """Test Zernike polynomial generation."""
    x, y = xy_grid
    result = zernike(x, y, 2, 2)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)
