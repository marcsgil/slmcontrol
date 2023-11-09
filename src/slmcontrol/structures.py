import numpy as np
from scipy import special
from scipy.special import factorial
from multimethod import multimethod
from slmcontrol.hologram import build_grid

@multimethod
def hg(x, y, m, n, w0):
    pm = special.hermite(m)
    pn = special.hermite(n)
    
    N =  np.sqrt(  2 / (np.pi * 2**( m+n ) * factorial(m) * factorial(n)) ) / w0

    return N * pm(np.sqrt(2) * x / w0) * pn(np.sqrt(2) * y / w0) * np.exp(-(x**2+y**2)/w0**2)

@multimethod
def hg(config_path: str, m, n, w0):
    x,y = build_grid(config_path)
    return hg(x, y, m, n, w0)

@multimethod
def lg(x, y, p, l, w0):
    lag = special.genlaguerre(p,abs(l))

    N = np.sqrt( 2 * factorial(p) / np.pi / factorial(p + np.abs(l))  ) / w0

    r = np.sqrt((x**2 + y**2)) / w0
    return (
        N * (np.sqrt(2)*r)**(abs(l)) 
        * np.exp(-r**2) * lag(2*r**2) 
        * np.exp(1j*l*np.arctan2(y,x)))

@multimethod
def lg(config_path: str, m, n, w0):
    x,y = build_grid(config_path)
    return lg(x, y, m, n, w0)

@multimethod
def diagonal_hg(x,y,m,n,w0, **kwargs):
    return hg((x-y)/np.sqrt(2),(x+y)/np.sqrt(2),m,n,w0, **kwargs)

@multimethod
def diagonal_hg(config_path: str, m, n, w0):
    x,y = build_grid(config_path)
    return diagonal_hg(x, y, m, n, w0)

@multimethod
def lens(x, y, fx, fy, lamb):
    k = 2*np.pi/lamb
    return np.exp(-1j*k*((x**2)/fx + (y**2)/fy))

@multimethod
def lens(config_path: str, fx, fy, lamb):
    x,y = build_grid(config_path)
    return lens(x, y, fx, fy, lamb)

@multimethod
def tilted_lens(x, y, f, theta, lamb):
    fx = f*np.cos(theta)
    fy = f/np.cos(theta)
    return lens(x, y, fx, fy, lamb)

@multimethod
def tilted_lens(config_path: str, f, theta, lamb):
    fx = f*np.cos(theta)
    fy = f/np.cos(theta)
    return tilted_lens(config_path, fx, fy, lamb)

@multimethod
def rectangular_apperture(x,y,a,b):
    return np.vectorize(lambda x,y: np.abs(x) <= a/2 and np.abs(y) <= b/2)(x,y)

@multimethod
def rectangular_apperture(config_path: str,a,b):
    x,y = build_grid(config_path)
    rectangular_apperture(x,y,a,b)

@multimethod
def square(x, y, l):
    return rectangular_apperture(x,y,l,l)

@multimethod
def square(config_path: str, l):
    x,y = build_grid(config_path)
    return square(x,y,l)

@multimethod
def single_slit(x,y,a):
    return rectangular_apperture(x,y,a,np.inf)

@multimethod
def single_slit(config_path: str,a):
    x,y = build_grid(config_path)
    return single_slit(x,y,a)

@multimethod
def double_slit(x,y,a,d):
    return rectangular_apperture(x - d/2,y,a,np.inf) + rectangular_apperture(x + d/2,y,a,np.inf)

@multimethod
def double_slit(config_path: str,a,d):
    x,y = build_grid(config_path)
    return double_slit(x,y,a,d)

@multimethod
def pupil(x,y,radius):
    return np.vectorize(lambda x,y: x**2+y**2 <= radius**2)(x,y)

@multimethod
def pupil(config_path: str, radius):
    x,y = build_grid(config_path)
    return pupil(x, y, radius)

@multimethod
def triangle(x, y, side_length):
    def is_inside(x,y):
        return y > -side_length/2/np.sqrt(3) and np.abs(x) < -y/np.sqrt(3) + side_length / 3
    return np.vectorize(is_inside)(x,y)

@multimethod
def triangle(config_path: str, side_length):
    x,y = build_grid(config_path)
    return triangle(x, y, side_length)