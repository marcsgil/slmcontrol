import unittest
import numpy as np
from slmcontrol.structures import (
    lg, hg, diagonal_hg, lens, tilted_lens, rectangular_aperture,
    square, single_slit, double_slit, pupil, triangle
)

class StructuresTestCase(unittest.TestCase):
    def test_structure_functions(self):
        """Test various structure generation functions."""
        x = np.random.rand(100)
        y = np.random.rand(200)
        test_shape = (200, 100)
        for function, args in [
            (lg, []),
            (hg, []),
            (diagonal_hg, [1, 1, 1]),
            (lens, [1.0, 1.0]),
            (tilted_lens, [1.0, 1.0]),
            (rectangular_aperture, [1.0, 1.0]),
            (square, [1.0]),
            (single_slit, [1.0]),
            (double_slit, [1.0, 1.0]),
            (pupil, [1.0]),
            (triangle, [1.0]),
        ]:
            result = function(x, y, *args)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, test_shape)
