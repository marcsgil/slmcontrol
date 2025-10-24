import unittest
import numpy as np
from slmcontrol.slm import SLMDisplay


class SLMTestCase(unittest.TestCase):
    def setUp(self):
        self.slm = SLMDisplay()

    def test_slm_updateArray(self):
        """Test hologram update functionality."""
        data = np.random.randint(
            0, 256, (self.slm.height, self.slm.width), dtype=np.uint8
        )
        self.slm.updateArray(data)
        unfit_data = np.random.randint(
            0, 256, (self.slm.height, self.slm.width + 1), dtype=np.uint8
        )
        with self.assertRaises(AssertionError):
            self.slm.updateArray(unfit_data)
        self.slm.close()

    def test_slm_initialization(self):
        """Test SLM multiple instance prevention."""
        with self.assertRaises(AssertionError):
            SLMDisplay()
        self.slm.close()

    def test_slm_close(self):
        """Test SLM close functionality."""
        self.slm.close()
        with self.assertRaises(AssertionError):
            self.slm.close()
