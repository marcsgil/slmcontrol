import numpy as np
import configparser
from multimethod import multimethod
from slmcontrol.inverse_functions import inverse_sinc,inverse_bessel0,inverse_bessel1
import slmcontrol

@multimethod
def build_grid(width, height, resX: int, resY: int, sparse=True):
    return np.meshgrid(
            np.linspace(-width/2 , width/2 ,resX),
            np.linspace(-height/2 ,height/2 ,resY),
            sparse=sparse)

@multimethod
def build_grid(config_path: str, sparse=True):
    config = configparser.ConfigParser()
    config.read(config_path)

    width = config['slm'].getfloat('width')
    height = config['slm'].getfloat('height')
    resX = config['slm'].getint('resX')
    resY = config['slm'].getint('resY')

    return build_grid(width, height, resX, resY, sparse=sparse)
    
    
def convert2angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def diffraction_grating(x,y,Lx,Ly):
    lx = Lx * (x[0,1] - x[0,0])
    ly = Ly * (y[1,0] - y[0,0])
    return convert2angle(2*np.pi*(x / lx + y / ly))

def normalize(holo,max):
    m = np.amin(holo)
    M = np.amax(holo)
    return np.round( max * (holo - m) / (M - m) ).astype('uint8')

def psi(phi,a,method='bessel1'):
    if method not in ("simple", "sinc", "bessel0" ,"bessel1"):
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
def generate_hologram(desired,input,x,y,max: int,xperiod,yperiod,xoffset,yoffset,method='bessel1'):

    _desired = np.roll(desired, (yoffset,xoffset), axis=(0,1))
    relative = _desired / input
    
    a = np.abs(relative)
    a /= np.max(a)

    lx = xperiod * (x[0,1] - x[0,0])
    ly = yperiod * (y[1,0] - y[0,0])
    phi = np.angle(relative) + 2 * np.pi * ( x/lx + y/ly )

    if method == 'bessel1':
        max *= 0.586

    return normalize(psi(phi,a,method), round(max))

@multimethod
def generate_hologram(desired,config_path: str,method='bessel1'):
    x,y = build_grid(config_path)

    config = configparser.ConfigParser()
    config.read(config_path)
    x,y = build_grid(config_path)

    max = config['slm'].getint('max')
    xoffset = config['input'].getint('xoffset')
    yoffset = config['input'].getint('yoffset')
    waist = config['input'].getfloat('waist')
    xperiod = config['grating'].getfloat('xperiod')
    yperiod = config['grating'].getfloat('yperiod')

    input =  slmcontrol.structures.hg(x,y,0,0,waist)
    
    return generate_hologram(desired,input,x,y,max,xperiod,yperiod,xoffset,yoffset,method=method)