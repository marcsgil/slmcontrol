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
        relative = desired / incoming

        result = generate_hologram(relative, 1, 1, 1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (slm.height, slm.width))

        slm.close()

    def test_zero_field_generates_finite_uniform_hologram(self):
        result = generate_hologram(
            np.zeros((4, 6), dtype=complex), 255, 10, 10
        )
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(np.all(result == result[0, 0]))

    def test_invalid_hologram_arguments(self):
        with self.assertRaises(ValueError):
            generate_hologram(np.ones(4), 255, 10, 10)
        with self.assertRaises(ValueError):
            generate_hologram(np.empty((0, 0)), 255, 10, 10)
        with self.assertRaises(ValueError):
            generate_hologram(np.ones((2, 2)), 255, 0, 10)
