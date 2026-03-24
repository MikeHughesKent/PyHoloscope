# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

This file contains numba-optimised functions relaatd to numrical refocusing.

"""

import math

import numpy as np

import numba
from numba import jit, njit

from pyholoscope.utils import dimensions
from pyholoscope.propagator import Propagator


@jit(nopython=True)
def propagator_numba(
    grid_size, wavelength, pixel_size, depth, geometry="plane", precision="single"
):
    """Numba optimised version of propagator().
    Creates Fourier domain propagator for angular spectrum method.
    Returns the propagator as a complex 2D numpy array. Generation is sped up
    by only calculating top left quadrant and then duplicating
    (with flips) to create the other quadrants.

    Note that 'precision' is currently not implemented in the numba version.

    Parameters:
         gridSize   : (int, int)
                      size of image (in pixels) to refocus, height x width
         pixelSize  : float
                      physical size of pixels
         wavelength : float
                      in same units as pixelSize
         depth      : float
                      refocus depth in same units as pixelSize

    Keyword Arguments:
         geometry   : str
                      'plane' (default) or 'point'
         precision  : str
                      numerical precision of ouptut, 'single' (defualt)
                      or 'double' [NOT IMPLEMENTED]
    """    

    grid_width = int(grid_size[1])
    grid_height = int(grid_size[0])

    width = float(grid_width) * float(pixel_size)
    height = float(grid_height) * float(pixel_size)

    centre_x = int(grid_width // 2)
    centre_y = int(grid_height // 2)

    prop_corner = np.zeros((centre_y + 1, centre_x + 1), dtype=numba.complex64)
    prop = np.zeros((grid_height, grid_width), dtype=numba.complex64)

    delta0x = 1 / width
    delta0y = 1 / height

    if geometry == "point":
        fac = math.pi * wavelength * depth

        for x in range(centre_x + 1):
            uSq = (delta0x * x) ** 2

            for y in range(centre_y + 1):
                vSq = (delta0y * y) ** 2
                phase = fac * (uSq + vSq)

                # This is about as twice as fast as using np.exp(1j * phase)
                prop_corner.real[y, x] = math.cos(phase)
                prop_corner.imag[y, x] = math.sin(phase)

    elif geometry == "plane":
        fac = -2 * math.pi * depth / wavelength
        for x in range(centre_x + 1):
            alphaSq = (float(wavelength) * x * delta0x) ** 2

            for y in range(centre_y + 1):
                betaSq = (float(wavelength) * y * delta0y) ** 2
                if alphaSq + betaSq < 1:
                    phase = fac * np.sqrt(1 - alphaSq - betaSq)

                    # This is about as twice as fast as using np.exp(1j * phase)
                    prop_corner.real[y, x] = math.cos(phase)
                    prop_corner.imag[y, x] = math.sin(phase)
    else:
        raise Exception("Invalid geometry.")

    # Duplicate the top left quadrant into the other three quadrants as
    # this is quicker then explicitly calculating the values
    prop[: centre_y + 1, : centre_x + 1] = prop_corner  # top left
    prop[: centre_y + 1, (centre_x + grid_width % 2):] = prop_corner[:, 1:][:, ::-1]
    prop[centre_y + grid_height % 2 :, :] = prop[1 : centre_y + 1, :][::-1, :]
    return prop
