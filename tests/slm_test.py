import pytest
import numpy as np
from juliacall import JuliaError
from slmcontrol.slm import SLMDisplay


def test_slm_initialization():
    """Test SLM initialization and multiple instance prevention."""
    slm = SLMDisplay()
    with pytest.raises(JuliaError):
        SLMDisplay()
    slm.close()


def test_slm_updateArray(slm):
    """Test hologram update functionality."""
    data = np.random.randint(0, 256, (slm.height, slm.width), dtype=np.uint8)
    slm.updateArray(data)

    unfit_data = np.random.randint(
        0, 256, (slm.height, slm.width + 1), dtype=np.uint8)
    with pytest.raises(JuliaError):
        slm.updateArray(unfit_data)


def test_slm_close_behavior(slm):
    """Test SLM closing behavior."""
    data = np.random.randint(0, 256, (slm.height, slm.width), dtype=np.uint8)
    slm.close()

    with pytest.raises(JuliaError):
        slm.updateArray(data)

    with pytest.raises(JuliaError):
        slm.close()
