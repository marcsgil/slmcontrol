import unittest
import numpy as np
from slmcontrol.hologram import generate_hologram
from slmcontrol.slm import SLMDisplay

class HologramTestCase(unittest.TestCase):
    def test_generate_hologram(self):
        """Test hologram generation."""
        slm = SLMDisplay()
        desired = np.random.randint(0, 256, (slm.height, slm.width), dtype=np.uint8)
        incoming = np.ones((slm.height, slm.width))

        result = generate_hologram(desired, incoming, 1.0, 1.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (slm.height, slm.width))

        slm.close()