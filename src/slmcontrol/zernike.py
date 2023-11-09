import numpy as np
from scipy.special import binom
import configparser
from slmcontrol.hologram import build_grid
from multimethod import multimethod

def R(r, n, m):
    s = 0.
    if (n-m) % 2 == 0:
        for k in range((n-m) // 2 + 1):
           s += (-1)**k * binom(n-k,k) * binom(n-2*k, (n-m) // 2 - k) * r**(n-2*k)
    return s

def zernike(x, y, n, m):
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    assert m in range(-n,n+1,2), "The index combination is does not define a valid polynomial"
    if m >= 0:
        return np.sqrt(2*(n+1))*R(r, n, m)*np.cos(m*phi)
    else:
        return np.sqrt(2*(n+1))*R(r, n, -m)*np.sin(m*phi)

@multimethod
def zernike_combination(x,y,indices,coefficients):
    p = np.zeros((y.shape[0], x.shape[1]))
    for c in range(len(coefficients)):
        p += coefficients[c]*zernike(x,y, indices[c][0], indices[c][1])
    return p

@multimethod
def zernike_combination(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    x,y = build_grid(config_path)

    indices = [[1,-1], [1, 1],    [2,-2], [2,0], [2, 2],    [3,-3], [3,-1], [3,1], [3,3]]
    coefficients = [config['zernike'].getfloat(key) for key in config['zernike']]

    return zernike_combination(x,y,indices,coefficients)