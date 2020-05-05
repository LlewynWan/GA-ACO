import numpy as np
from random import randint
from scipy.signal import convolve2d

def randfunc(x: np.array, y: np.array):
    z1 = convolve2d(10*randint(1,10)*np.sin(5*randint(1,10)*x+0.1*randint(-10,10)),
            10*randint(1,10)*np.cos(5*randint(1,10)*x+0.1*randint(-10,10)), 'same')
    z2 = convolve2d(10*randint(1,10)*np.sin(5*randint(1,10)*y+0.1*randint(-10,10)),
            10*randint(1,10)*np.cos(5*randint(1,10)*y+0.1*randint(-10,10)), 'same')
    z = convolve2d(z1, z2, 'same')
    z += 0.1 * randint(1,10) * np.transpose(z)
    return z

