import numpy as np
import configparser
from multimethod import multimethod
from slmcontrol.inverse_functions import inverse_sinc, inverse_bessel0, inverse_bessel1
import slmcontrol


@multimethod
def build_grid(config_path: str, nmasks=1, sparse=True):
    """Constructs a 2D meshgrid.

    Args:
        config_path (str): path for the configuration file of the SLM.
        nmasks (int, optional): The number of masks that will be shown simutaneously in the SLM. Defaults to 1.
        sparse (bool, optional): whether or not the grid should be sparse. Defaults to True.

    Returns:
        (tuple[array_like,array_like]): x and y meshgrids
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    width = config['slm'].getfloat('width')
    height = config['slm'].getfloat('height')
    resX = config['slm'].getint('resX')
    resY = config['slm'].getint('resY')

    return build_grid(width, height, resX, resY, nmasks=nmasks, sparse=sparse)


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
        (tuple[array_like,array_like]): x and y meshgrids

    Note:
        There is also a method ` build_grid(config_path, nmasks=1, sparse=True)` where `config_path` is the path for the configuration file.
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


def convert2angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def normalize(holo, max_modulation):
    m = np.amin(holo)
    M = np.amax(holo)
    return np.round(max_modulation * (holo - m) / (M - m)).astype('uint8')


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
def generate_hologram(desired, config_path: str, method='bessel1'):
    """Generates a hologram to be displayed in the SLM.

    Args:
        desired (array_like | tuple | list): the (list of) field(s) that one wishes to produce
        config_path (str): path for the configuration file of the SLM.
        method (str, optional): Algorithm to be used in the generation of the hologram. 
        Possible values are:

            - 'simple': Method A of reference [2] 

            - 'sinc': Type 1 of reference [1] or method C of reference [2] 

            - 'bessel0': Type 2 of reference [1] 

            - 'bessel1': Type 3 of reference [1] or method F of reference [2]


        We recomend 'bessel1' for the best beam quality, or 'sinc' for the most power.
        Defaults to 'bessel1'.

    Returns:
        (array_like): Hologram ready to be sent to the SLM.

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

    max_modulation = config['slm'].getint('max_modulation')
    xoffset = config['incoming'].getint('xoffset')
    yoffset = config['incoming'].getint('yoffset')
    waist = config['incoming'].getfloat('waist')
    xperiod = config['grating'].getfloat('xperiod')
    yperiod = config['grating'].getfloat('yperiod')

    incoming = slmcontrol.structures.hg(x, y, 0, 0, waist)
    return generate_hologram(desired, incoming, x, y, max_modulation, xperiod, yperiod, xoffset, yoffset, method=method)


@multimethod
def generate_hologram(desired: tuple | list, incoming, x, y, max_modulation: int, xperiod, yperiod, xoffset, yoffset, method='bessel1'):
    holos = [generate_hologram(d, incoming, x, y, max_modulation, xperiod,
                               yperiod, xoffset, yoffset, method=method) for d in desired]

    N = int(np.sqrt(len(holos)))
    if N == np.sqrt(len(holos)):
        return np.concatenate([np.concatenate(holos[i:i+N], axis=1) for i in range(0, N**2, N)], axis=0)
    else:
        return np.concatenate(holos, axis=1)


@multimethod
def generate_hologram(desired, incoming, x, y, max_modulation: int, xperiod, yperiod, xoffset, yoffset, method='bessel1'):
    r"""Generates a hologram to be displayed in the SLM.

    Args:
        desired (array_like | tuple | list): the (list of) field(s) that one wishes to produce
        incoming (array_like): field that arrives at the SLM
        x (array_like): x grid
        y (array_like): y grid
        max_modulation (int): Maximum modulated value, which sould correspond to a phase of \(2 \pi\). Depends on the SLM specifications
        xperiod (Real): Period (in pixels) of the diffraction grating in the x direction
        yperiod (Real): Period (in pixels) of the diffraction grating in the y direction
        xoffset (int): Translation (in pixels), of the input beam in the x direction, with respect to the origin
        yoffset (int): Translation (in pixels), of the input beam in the y direction, with respect to the origin
        method (str, optional): Algorithm to be used in the generation of the hologram. 
            Possible values are:

                1. 'simple': Method A of reference [2] 

                2. 'sinc': Type 1 of reference [1] or method C of reference [2] 

                3. 'bessel0': Type 2 of reference [1] 

                4. 'bessel1': Type 3 of reference [1] or method F of reference [2] 

            We recomend 'bessel1' for the best beam quality, or 'sinc' for the most power.
            Defaults to 'bessel1'.

    Returns:
        (array_like): Hologram ready to be sent to the SLM.

    Note:
        There is also a method `generate_hologram(desired, config_path, method='bessel1')` where `config_path` is the path for the configuration file.

    References:
        [1] Victor Arrizón, Ulises Ruiz, Rosibel Carrada, and Luis A. González, 
            "Pixelated phase computer holograms for the accurate encoding of scalar complex fields," 
            J. Opt. Soc. Am. A 24, 3500-3507 (2007)

        [2] Thomas W. Clark, Rachel F. Offer, Sonja Franke-Arnold, Aidan S. Arnold, and Neal Radwell, 
            "Comparison of beam generation techniques using a phase only spatial light modulator," 
            Opt. Express 24, 6249-6264 (2016)
    """

    relative = desired / np.roll(incoming, (yoffset, xoffset), axis=(0, 1))

    a = np.abs(relative)
    a /= np.max(a)

    lx = xperiod * (x[0, 1] - x[0, 0])
    ly = yperiod * (y[1, 0] - y[0, 0])
    phi = np.angle(relative) + 2 * np.pi * (x/lx + y/ly)

    if method == 'bessel1':
        max_modulation *= 0.586

    return normalize(psi(phi, a, method), round(max_modulation))
