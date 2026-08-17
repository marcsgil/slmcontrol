import unittest
import numpy as np
from scipy.special import j0
from slmcontrol.hologram import generate_hologram, inv_j0
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

    def test_inverse_j0_uses_first_monotonic_branch(self):
        amplitudes = np.linspace(0, 1, 21)
        inverse = inv_j0(amplitudes)

        self.assertTrue(np.all(np.diff(inverse) <= 0))
        np.testing.assert_allclose(j0(inverse), amplitudes, atol=1e-6)
        self.assertAlmostEqual(inverse[0], 2.404825557695773)
        self.assertAlmostEqual(inverse[-1], 0)

    def test_besselj0_unit_amplitude_reduces_to_carrier(self):
        result = generate_hologram(
            np.ones((1, 4), dtype=complex), 255, 4, 10, method="BesselJ0"
        )

        expected = np.array([[0, 64, 128, 191]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_besselj0_implements_type_2_phase(self):
        relative = np.array([[1, j0(1), 0]], dtype=complex)
        result = generate_hologram(
            relative, 255, 4, 10, method="BesselJ0"
        )

        expected_phase = np.array([[0, np.pi / 2 + 1, np.pi]])
        expected = np.astype(
            np.round(255 * expected_phase / (2 * np.pi)), np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_besselj0_zero_field_is_finite(self):
        result = generate_hologram(
            np.zeros((4, 6), dtype=complex), 255, 10, 10, method="BesselJ0"
        )

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_invalid_hologram_arguments(self):
        with self.assertRaises(ValueError):
            generate_hologram(np.ones(4), 255, 10, 10)
        with self.assertRaises(ValueError):
            generate_hologram(np.empty((0, 0)), 255, 10, 10)
        with self.assertRaises(ValueError):
            generate_hologram(np.ones((2, 2)), 255, 0, 10)
