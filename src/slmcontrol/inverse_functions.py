from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
from numpy import sinc
from scipy.special import jv
import numpy as np

def get_inverse_function(f,domain,bracket,N=128):
    def inverse(x):
        if x<domain[0] or x> domain[1]:
            raise ValueError('You tried to calculate the inverse function outside of its domain.')
        
        if x==domain[0]:
            return float(bracket[0])
        elif x==domain[1]:
            return float(bracket[1])
        else:
            return root_scalar(lambda y: f(y) - x, bracket=bracket).root
        
    vectorized_inverse = np.vectorize(inverse)

    x = np.linspace(domain[0],domain[1],N)
    y = vectorized_inverse(x)

    spline = CubicSpline(x,y,extrapolate=False)

    def interpolated_inverse(x):
        if np.any(x<domain[0]) or np.any(x>domain[1]):
            raise ValueError('You tried to calculate the inverse function outside of its domain.')
        else:
            return spline(x)

    return interpolated_inverse

inverse_sinc = get_inverse_function(np.sinc,[0,1],[1,0])
inverse_bessel0 = get_inverse_function(lambda x: jv(0,x),[0,1], [2.4047, 0.])
inverse_bessel1 = get_inverse_function(lambda x: jv(1,x),[0.,0.5819],[0, 1.84])