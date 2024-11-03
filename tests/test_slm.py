import pytest
import numpy as np
from juliacall import JuliaError
from slmcontrol.slm import SLM

def test_slm_initialization():
    """Test SLM initialization and multiple instance prevention."""
    slm = SLM()
    with pytest.raises(JuliaError):
        SLM()
    slm.close()

def test_slm_update_hologram(slm):
    """Test hologram update functionality."""
    data = np.random.randint(0, 256, (slm.width, slm.height), dtype=np.uint8)
    slm.update_hologram(data)

    unfit_data = np.random.randint(0, 256, (slm.width + 1, slm.height), dtype=np.uint8)
    with pytest.raises(JuliaError):
        slm.update_hologram(unfit_data)

def test_slm_close_behavior(slm):
    """Test SLM closing behavior."""
    data = np.random.randint(0, 256, (slm.width, slm.height), dtype=np.uint8)
    slm.close()
    
    with pytest.raises(JuliaError):
        slm.update_hologram(data)
    
    with pytest.raises(JuliaError):
        slm.close()

