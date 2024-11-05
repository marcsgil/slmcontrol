import pytest
import numpy as np
from slmcontrol.slm import SLMDisplay
from juliacall import JuliaError


@pytest.fixture
def test_shape():
    """Fixture providing a test shape."""
    width = 20
    height = 10
    return height, width


@pytest.fixture
def xy_grid(test_shape):
    """Fixture providing standard x and y coordinate grids."""
    height, width = test_shape
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    return x, y


@pytest.fixture
def slm():
    """Fixture providing an SLM instance with cleanup."""
    device = SLMDisplay()
    yield device
    try:
        device.close()
    except JuliaError:
        pass


@pytest.fixture
def random_array(test_shape):
    """Fixture providing a random 10x10 array."""
    return np.random.rand(*test_shape)
