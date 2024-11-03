import pytest
import numpy as np
from slmcontrol.hologram import generate_hologram


def test_generate_hologram(xy_grid, random_array):
    """Test hologram generation."""
    x, y = xy_grid
    desired = random_array
    incoming = random_array

    result = generate_hologram(desired, incoming, x, y, 1.0, 1.0, 1.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)
