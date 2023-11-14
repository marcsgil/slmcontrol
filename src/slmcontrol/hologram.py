import numpy as np
import configparser
from multimethod import multimethod
from slmcontrol.inverse_functions import inverse_sinc, inverse_bessel0, inverse_bessel1
import slmcontrol


def read_pars(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    max = config['slm'].getint('max')
    xoffset = config['input'].getint('xoffset')
    yoffset = config['input'].getint('yoffset')
    waist = config['input'].getfloat('waist')
    xperiod = config['grating'].getfloat('xperiod')
    yperiod = config['grating'].getfloat('yperiod')

    return waist, max, xperiod, yperiod, xoffset, yoffset


@multimethod
def build_grid(width, height, resX: int, resY: int, nmasks=1, sparse=True):
    """Constructs a 2D meshgrid.

    Args:
        width (Real): width of the grid. The units are arbitrary, but the same for width and height.
        height (Real): height of the grid. The units are arbitrary, but the same for width and height.
        resX (int): number of points in the x direction
        resY (int): number of points in the y direction
        nmasks (int, optional): The number of masks that will be shown simutaneously in the SLM. Defaults to 1.
        sparse (bool, optional): whether or not the grid should be sparse. Defaults to True.

    Returns:
        tuple[array_like,array_like]: x and y meshgrids
    """

    N = int(np.sqrt(nmasks))
    if N == np.sqrt(nmasks):
        return np.meshgrid(
            np.linspace(-width/(2*N), width/(2*N), resX // N),
            np.linspace(-height/(2*N), height/(2*N), resY // N),
            sparse=sparse)
    else:
        return np.meshgrid(
            np.linspace(-width/(2*nmasks), width/(2*nmasks), resX // nmasks),
            np.linspace(-height/2, height/2, resY),
            sparse=sparse)


@multimethod
def build_grid(config_path: str, nmasks=1, sparse=True):
    """Constructs a 2D meshgrid.

    Args:
        config_path (str): path for the configuration file of the SLM.
        nmasks (int, optional): The number of masks that will be shown simutaneously in the SLM. Defaults to 1.
        sparse (bool, optional): whether or not the grid should be sparse. Defaults to True.

    Returns:
        tuple[array_like,array_like]: x and y meshgrids
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    width = config['slm'].getfloat('width')
    height = config['slm'].getfloat('height')
    resX = config['slm'].getint('resX')
    resY = config['slm'].getint('resY')

    return build_grid(width, height, resX, resY, nmasks=nmasks, sparse=sparse)


def convert2angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def normalize(holo, max):
    m = np.amin(holo)
    M = np.amax(holo)
    return np.round(max * (holo - m) / (M - m)).astype('uint8')


def psi(phi, a, method='bessel1'):
    if method not in ("simple", "sinc", "bessel0", "bessel1"):
        raise ValueError(
            "Known methods are 'simple', 'sinc', 'bessel0', 'bessel1'. Got %s." % method
        )

    if method == 'simple':
        return a * convert2angle(phi)
    elif method == 'sinc':
        return (1-inverse_sinc(a)) * convert2angle(phi)
    elif method == 'bessel0':
        return convert2angle(phi + inverse_bessel0(a) * np.sin(phi))
    elif method == 'bessel1':
        return inverse_bessel1(a * 0.5818) * np.sin(phi)


@multimethod
def generate_hologram(desired, input, x, y, max: int, xperiod, yperiod, xoffset, yoffset, method='bessel1'):
    """Generates a hologram to be displayed in the SLM.

    Args:
        desired (array_like | tuple | list): the (list of) field(s) that one wishes to produce
        input (array_like): field that arrives at the SLM
        x (array_like): x grid
        y (array_like): y grid
        max (int): Maximum modulated value, which sould correspond to a phase of 2pi. Depends on the SLM specifications
        xperiod (Real): Period (in pixels) of the diffraction grating in the x direction
        yperiod (Real): Period (in pixels) of the diffraction grating in the y direction
        xoffset (int): Translation (in pixels), of the output beam in the x direction
        yoffset (int): Translation (in pixels), of the output beam in the y direction
        method (str, optional): Algorithm to be used in the generation of the hologram. 
            Possible values are:
                'simple': Method A of reference [2]
                'sinc': Type 1 of reference [1] or method C of reference [2]
                'bessel0': Type 2 of reference [1]
                'bessel1': Type 3 of reference [1] or method F of reference [2]

        We recomend 'bessel1' for the best beam quality, or 'sinc' for the most power.
        Defaults to 'bessel1'.

    Returns:
        array_like: Hologram ready to be sent to the SLM.

    References:
        [1] Victor Arrizón, Ulises Ruiz, Rosibel Carrada, and Luis A. González, 
            "Pixelated phase computer holograms for the accurate encoding of scalar complex fields," 
            J. Opt. Soc. Am. A 24, 3500-3507 (2007)

        [2] Thomas W. Clark, Rachel F. Offer, Sonja Franke-Arnold, Aidan S. Arnold, and Neal Radwell, 
            "Comparison of beam generation techniques using a phase only spatial light modulator," 
            Opt. Express 24, 6249-6264 (2016)
    """

    _desired = np.roll(desired, (yoffset, xoffset), axis=(0, 1))
    relative = _desired / input

    a = np.abs(relative)
    a /= np.max(a)

    lx = xperiod * (x[0, 1] - x[0, 0])
    ly = yperiod * (y[1, 0] - y[0, 0])
    phi = np.angle(relative) + 2 * np.pi * (x/lx + y/ly)

    if method == 'bessel1':
        max *= 0.586

    return normalize(psi(phi, a, method), round(max))


@multimethod
def generate_hologram(desired: tuple | list, input, x, y, max: int, xperiod, yperiod, xoffset, yoffset, method='bessel1'):
    holos = [generate_hologram(d, input, x, y, max, xperiod,
                               yperiod, xoffset, yoffset, method=method) for d in desired]

    N = int(np.sqrt(len(holos)))
    if N == np.sqrt(len(holos)):
        return np.concatenate([np.concatenate(holos[i:i+N], axis=1) for i in range(0, N**2, N)], axis=0)
    else:
        return np.concatenate(holos, axis=1)


@multimethod
def generate_hologram(desired, config_path: str, method='bessel1'):
    """Generates a hologram to be displayed in the SLM.

    Args:
        desired (array_like | tuple | list): the (list of) field(s) that one wishes to produce
        config_path (str): path for the configuration file of the SLM.
        method (str, optional): Algorithm to be used in the generation of the hologram. 
            Possible values are:
                'simple': Method A of reference [2]
                'sinc': Type 1 of reference [1] or method C of reference [2]
                'bessel0': Type 2 of reference [1]
                'bessel1': Type 3 of reference [1] or method F of reference [2]

        We recomend 'bessel1' for the best beam quality, or 'sinc' for the most power.
        Defaults to 'bessel1'.

    Returns:
        array_like: Hologram ready to be sent to the SLM.

    References:
        [1] Victor Arrizón, Ulises Ruiz, Rosibel Carrada, and Luis A. González, 
            "Pixelated phase computer holograms for the accurate encoding of scalar complex fields," 
            J. Opt. Soc. Am. A 24, 3500-3507 (2007)

        [2] Thomas W. Clark, Rachel F. Offer, Sonja Franke-Arnold, Aidan S. Arnold, and Neal Radwell, 
            "Comparison of beam generation techniques using a phase only spatial light modulator," 
            Opt. Express 24, 6249-6264 (2016)
    """
    T = type(desired)
    if T == tuple or T == list:
        nmasks = len(desired)
    else:
        nmasks = 1

    x, y = build_grid(config_path, nmasks=nmasks)

    config = configparser.ConfigParser()
    config.read(config_path)

    max = config['slm'].getint('max')
    xoffset = config['input'].getint('xoffset')
    yoffset = config['input'].getint('yoffset')
    waist = config['input'].getfloat('waist')
    xperiod = config['grating'].getfloat('xperiod')
    yperiod = config['grating'].getfloat('yperiod')

    input = slmcontrol.structures.hg(x, y, 0, 0, waist)
    return generate_hologram(desired, input, x, y, max, xperiod, yperiod, xoffset, yoffset, method=method)
