import numpy as np
from slmcontrol.structures import *


def test_hg():
    assert np.isclose(hg(1, 1, 0, 0, 1), 0.10798193302637613)
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    assert hg(x, y, 0, 0, 1).shape == (3, 5)


def test_lg():
    assert np.isclose(lg(1, 1, 0, 0, 1), 0.10798193302637613)
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    assert lg(x, y, 0, 0, 1).shape == (3, 5)


def test_diagonal_hg():
    assert np.isclose(diagonal_hg(1, 1, 0, 0, 1), 0.10798193302637613)
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    assert diagonal_hg(x, y, 0, 0, 1).shape == (3, 5)


def test_lens():
    assert np.isclose(lens(0, 0, 1, 1, 1), 1)
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    assert lens(x, y, 1, 1, 1).shape == (3, 5)


def test_tilted_lens():
    assert np.isclose(tilted_lens(0, 0, 1, 1, 1), 1)
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    assert tilted_lens(x, y, 1, 1, 1).shape == (3, 5)


def test_rectangular_apperture():
    assert rectangular_apperture(0, 0, 1, 1)
    x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    assert np.all(rectangular_apperture(x, y, 2, 2))


def test_square():
    assert square(0, 0, 1)
    x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    assert np.all(square(x, y, 2))


def test_single_slit():
    assert single_slit(0, 0, 1)
    x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    assert np.all(single_slit(x, y, 2))


def test_double_slit():
    assert not double_slit(0, 0, 1, 2)
    assert double_slit(0, 0, 1, 1)
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    assert double_slit(x, y, 1, 2).shape == (3, 5)


def test_pupil():
    assert pupil(0, 0, 1)
    x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    assert np.all(pupil(x, y, 2))


def test_triangle():
    assert triangle(0, 0, 1)
    assert not triangle(1, 1, 1)
    x, y = np.meshgrid(np.arange(3), np.arange(5))
    assert triangle(x, y, 1).shape == (5, 3)


def test_linear_combination():
    assert linear_combination([1, 2], [1, 2]) == 5
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    for order in range(1, 6):
        for k in range(order+1):
            true_result = diagonal_hg(x, y, order-k, k, 1)
            coeffs = [b(order-m, m, k) for m in range(order+1)]
            result = linear_combination('hg', coeffs, x, y, 1)
            assert np.allclose(result, true_result)


def test_b():
    assert b(0, 0, 0) == 1

    assert np.allclose(b(1, 0, 0), 1/np.sqrt(2))
    assert np.allclose(b(0, 1, 0), 1/np.sqrt(2))
    assert np.allclose(b(1, 0, 1), 1/np.sqrt(2))
    assert np.allclose(b(0, 1, 1), -1/np.sqrt(2))

    assert np.allclose(b(0, 2, 0), 1/2)
    assert np.allclose(b(0, 2, 1), -1/np.sqrt(2))
    assert np.allclose(b(0, 2, 2), 1/2)
    assert np.allclose(b(1, 1, 0), 1/np.sqrt(2))
    assert np.allclose(b(1, 1, 1), 0)
    assert np.allclose(b(1, 1, 2), -1/np.sqrt(2))
    assert np.allclose(b(2, 0, 0), 1/2)
    assert np.allclose(b(2, 0, 1), 1/np.sqrt(2))
    assert np.allclose(b(2, 0, 2), 1/2)
