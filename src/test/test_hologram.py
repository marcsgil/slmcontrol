import numpy as np
from slmcontrol.hologram import build_grid, generate_hologram
from pathlib import Path
import configparser

def test_build_grid():
    width, height, resX, resY = 10, 10, 100, 100
    x, y = build_grid(width, height, resX, resY)

    assert x.shape == (1, resX)
    assert y.shape == (resY, 1)
    assert np.allclose(x[0, 0], -width / 2)
    assert np.allclose(y[0, 0], -height / 2)

def test_build_grid_with_nmasks():
    width, height, resX, resY, nmasks = 10, 10, 100, 100, 4
    N = np.sqrt(nmasks)
    x, y = build_grid(width, height, resX, resY, nmasks)

    assert x.shape == (1, resX // N)
    assert y.shape == (resY // N, 1)
    assert np.allclose(x[0, 0], -width / (2 * N))
    assert np.allclose(y[0, 0], -height / (2 * N))

def test_build_grid_with_config_file(tmp_path):
    # Create a temporary .ini file
    config_path = tmp_path / "config.ini"
    config = configparser.ConfigParser()

    config['slm'] = {
        'width': '10',
        'height': '10',
        'resX': '100',
        'resY': '100'
    }

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    x, y = build_grid(str(config_path))

    assert x.shape == (1, 100)
    assert y.shape == (100, 1)
    assert np.allclose(x[0, 0], -5)
    assert np.allclose(y[0, 0], -5)

def test_generate_hologram():
    desired = np.ones((100, 100))
    incoming = np.ones((100, 100))
    x, y = build_grid(10, 10, 100, 100)
    max_modulation = 255
    xperiod = 10
    yperiod = 10
    xoffset = 0
    yoffset = 0
    method = 'bessel1'
    for method in ('simple', 'sinc', 'bessel0', 'bessel1'):
        hologram = generate_hologram(desired, incoming, x, y, max_modulation, xperiod, yperiod, xoffset, yoffset, method)
        assert hologram.shape == (100, 100)

def test_generate_hologram_with_list():
    desired = [np.ones((100, 100)), np.ones((100, 100))]
    incoming = np.ones((100, 100))
    x, y = build_grid(10, 10, 100, 100)
    max_modulation = 255
    xperiod = 10
    yperiod = 10
    xoffset = 0
    yoffset = 0

    for method in ('simple', 'sinc', 'bessel0', 'bessel1'):
        hologram = generate_hologram(desired, incoming, x, y, max_modulation, xperiod, yperiod, xoffset, yoffset, method)
        assert hologram.shape == (100, 200)

def test_generate_hologram_with_config_file(tmp_path):
    # Create a temporary .ini file
    config_path = tmp_path / "config.ini"
    config = configparser.ConfigParser()

    config['slm'] = {
        'width': '10',
        'height': '10',
        'resX': '100',
        'resY': '100',
        'max_modulation': '255'
    }

    config['incoming'] = {
        'xoffset': '0',
        'yoffset': '0',
        'waist': '1'
    }

    config['grating'] = {
        'xperiod': '10',
        'yperiod': '10'
    }

    with open(config_path, 'w') as configfile:
        config.write(configfile)

    desired = np.ones((100, 100))
    method = 'bessel1'

    hologram = generate_hologram(str(config_path), desired, method)

    assert hologram.shape == (100, 100)