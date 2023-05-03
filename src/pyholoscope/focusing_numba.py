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
def propagator_numba(gridSize, wavelength, pixelSize, depth, geometry = 'plane'):
    """ Creates Fourier domain propagator for angular spectrum meethod. GridSize
    is size of image (in pixels) to refocus (must be multiple of 2). 
    Uses Numba and quadrant method for speed.
    """
    assert gridSize % 2 == 0, "Grid size must be even"
    
    area = gridSize * pixelSize
    propCorner = np.zeros((int(gridSize/2) + 1, int(gridSize/2) + 1), dtype = 'complex64')
    prop = np.zeros((gridSize, gridSize), dtype = 'complex64')

    delta0 = 1/area
    midPoint = int(gridSize / 2)
    
    if geometry == 'point':
        fac = math.pi*wavelength*depth

        for x in range(int(gridSize/2) + 1) :
             uSq = (delta0*x)**2

             for y in range(int(gridSize/2) + 1) :
            
                 vSq = (delta0*y)**2

                 phase = fac*(uSq + vSq)

                 # This is about as twice as fast as using np.exp(1j * phase)
                 propCorner.real[y,x] = math.cos(phase)
                 propCorner.imag[y,x] = math.sin(phase)
        
    elif geometry == 'plane':   
        fac = 2 * math.pi * depth / wavelength
        for x in range(int(gridSize/2) + 1) :
             alphaSq = (wavelength * x * delta0)**2

             for y in range(int(gridSize/2) + 1) :
                 betaSq = (wavelength * y * delta0)**2
                 if alphaSq + betaSq < 1:
                     phase = fac * np.sqrt(1 - alphaSq - betaSq) 
                     
                     # This is about as twice as fast as using np.exp(1j * phase)
                     propCorner.real[y,x] = math.cos(phase)
                     propCorner.imag[y,x] = math.sin(phase)
    else:
        raise Exception("Invalid geometry.")               
            
      
    # Duplicate the top left quadrant into the other three quadrants as
    # this is quicker then explicitly calculating the values
    prop[:midPoint + 1, :midPoint + 1] = propCorner                  # top left
    prop[:midPoint + 1, midPoint:] = (np.fliplr(propCorner[:, 1:]) )     # top right
    prop[midPoint:, :] = np.flipud(prop[1:midPoint + 1, :])  # bottom left

   
    return prop
