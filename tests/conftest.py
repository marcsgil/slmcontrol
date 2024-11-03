import pytest
import numpy as np
from slmcontrol.slm import SLM
from juliacall import JuliaError
import os
os.environ["DISPLAY"] = ":0"


@pytest.fixture
def xy_grid():
    """Fixture providing standard x and y coordinate grids."""
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    return x, y


@pytest.fixture
def slm():
    """Fixture providing an SLM instance with cleanup."""
    device = SLM()
    yield device
    try:
        device.close()
    except JuliaError:
        pass


@pytest.fixture
def random_array():
    """Fixture providing a random 10x10 array."""
    return np.random.rand(10, 10)
