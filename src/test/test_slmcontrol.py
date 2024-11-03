from juliacall import JuliaError
from slmcontrol.zernike import zernike
from slmcontrol.structures import lg, hg, diagonal_hg, lens, tilted_lens, rectangular_apperture, square, single_slit, double_slit, pupil, triangle
from slmcontrol.hologram import generate_hologram
from slmcontrol.slm import SLM
import numpy as np
import unittest


class TestSLMControl(unittest.TestCase):

    def test_slm(self):
        slm = SLM()
        self.assertRaises(JuliaError, SLM)
        data = np.random.randint(
            0, 256, (slm.width, slm.height), dtype=np.uint8)
        slm.update_hologram(data)
        unfit_data = np.random.randint(
            0, 256, (slm.width + 1, slm.height), dtype=np.uint8)
        self.assertRaises(JuliaError, slm.update_hologram, unfit_data)
        slm.close()
        self.assertRaises(JuliaError, slm.update_hologram, data)

    def test_slm2(self):
        slm = SLM()
        slm.close()
        self.assertRaises(JuliaError, slm.close)

    def test_generate_hologram(self):
        desired = np.random.rand(10, 10)
        incoming = np.random.rand(10, 10)
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = generate_hologram(desired, incoming, x, y, 1.0, 1.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_lg(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = lg(x, y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_hg(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = hg(x, y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_diagonal_hg(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = diagonal_hg(x, y, 1, 1, 1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_lens(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = lens(x, y, 1.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_tilted_lens(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = tilted_lens(x, y, 1.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_rectangular_apperture(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = rectangular_apperture(x, y, 1.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_square(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = square(x, y, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_single_slit(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = single_slit(x, y, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_double_slit(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = double_slit(x, y, 1.0, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_pupil(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = pupil(x, y, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_triangle(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = triangle(x, y, 1.0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))

    def test_zernike(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        result = zernike(x, y, 2, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))


if __name__ == '__main__':
    unittest.main()
