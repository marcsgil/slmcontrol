import unittest
from slmcontrol.zernike import zernike
import numpy as np

class ZernikeTestCase(unittest.TestCase):
    def test_zernike(self):
        """Test Zernike polynomial generation."""
        x = np.random.rand(100)
        y = np.random.rand(200)
        test_shape = (200, 100)
        result = zernike(x, y, 2, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_shape)
