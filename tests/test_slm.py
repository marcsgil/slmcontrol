import unittest
import numpy as np
from juliacall import JuliaError
from slmcontrol.slm import SLMDisplay

class SLMTestCase(unittest.TestCase):
    def setUp(self):
        slm = SLMDisplay()
        self.data = np.random.randint(0, 256, (slm.height, slm.width), dtype=np.uint8)
        slm.close()

    def test_slm_initialization(self):
        """Test SLM initialization and multiple instance prevention."""
        slm = SLMDisplay()
        with self.assertRaises(JuliaError):
            SLMDisplay()
        slm.close()

    def test_slm_updateArray(self):
        """Test hologram update functionality."""
        slm = SLMDisplay()
        slm.updateArray(self.data)
        unfit_data = np.random.randint(0, 256, (slm.height, slm.width + 1), dtype=np.uint8)
        with self.assertRaises(JuliaError):
            slm.updateArray(unfit_data)
        slm.close()

    def test_slm_close_behavior(self):
        """Test SLM closing behavior."""
        slm = SLMDisplay()
        slm.close()
        with self.assertRaises(JuliaError):
            slm.updateArray(self.data)
        with self.assertRaises(JuliaError):
            slm.close()