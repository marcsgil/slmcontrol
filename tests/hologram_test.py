import pytest
import numpy as np
from slmcontrol.hologram import generate_hologram


def test_generate_hologram(random_array, test_shape):
    """Test hologram generation."""
    desired = random_array
    incoming = random_array

    result = generate_hologram(desired, incoming, 1.0, 1.0, 1.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == test_shape
