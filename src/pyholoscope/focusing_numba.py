# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Mike Hughes, Applied Optics Group, University of Kent

PyHoloscope is a python library for holographic microscopy.

This file contains numba-optimised functions relaatd to numrical refocusing.

"""
import math
import cmath
import numpy as np


from numba import jit, njit

@jit(nopython = True)   
def propagator_numba(gridSize, wavelength, pixelSize, depth):
    """ Creates Fourier domain propagator for angular spectrum meethod. GridSize
    is size of image (in pixels) to refocus (must be multiple of 2). 
    Uses Numba and quadrant method for speed.
    """
    assert gridSize % 2 == 0, "Grid size must be even"

    area = gridSize * pixelSize
    propCorner = np.zeros((int(gridSize/2), int(gridSize/2)), dtype = 'complex64')
    delta0 = 1/area
    midPoint = int(gridSize / 2)
    fac = math.pi*wavelength*depth
    
    for x in range(gridSize/2):
        uSq = (delta0*(x - gridSize/2 +.5))**2

        for y in range(gridSize/2):
        
            vSq = (delta0*(y - gridSize/2 +.5))**2

            phase = fac*(uSq + vSq)

            # This is about as twice as fast as using np.exp(1j * phase)
            propCorner.real[x,y] = math.cos(phase)
            propCorner.imag[x,y] = math.sin(phase)
            
            
    # Copy the top left qaurter into other quadrants (with flips) to buid
    # the whole propagator        
    prop = np.zeros((gridSize, gridSize), dtype ='complex64')
    prop[:midPoint, :midPoint] = propCorner
    prop[midPoint:, :midPoint] = np.flipud(propCorner)
    prop[:, midPoint:] = np.fliplr(prop[:, :midPoint])

    
    return prop
