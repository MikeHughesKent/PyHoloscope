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
import numba
from numba import jit, njit
from pyholoscope.utils import dimensions

@jit(nopython = True)   
def propagator_numba(gridSize, wavelength, pixelSize, depth, geometry = 'plane', precision = 'single'):
    """ Creates Fourier domain propagator for angular spectrum method. Speeds
    up generation by only calculating top left quadrant and then duplicating 
    (with flips) to create the other quadrants. Returns the propagator as a 
    complex 2D numpy array. Uses Numba JIT, typically several times faster
    than calling propagator().
    
    Arguments:
        gridSize   : tuple of (float, float), size of image.
        pixelSize  : float, physical size of pixels
        wavelength : float, in same units as pixelSize
        depth      : float, refocus depth in same units as pixelSize
    Optional Keyword Arguments:
        geometry   : str, 'plane' (defualt) or 'point'
    
        
    NOTE: precision is currently not implemented, propagator numba will always
          use single precision

    
    """
    #assert gridSize % 2 == 0, "Grid size must be even"
    
   
    gridWidth = int(gridSize[0])
    gridHeight = int(gridSize[1])
  
    
    width = float(gridWidth) * float(pixelSize)
    height = float(gridHeight) * float(pixelSize)
    
    centreX = int(gridWidth//2)
    centreY = int(gridHeight//2)

    propCorner = np.zeros((centreY + 1, centreX + 1), dtype = numba.complex64)
    prop = np.zeros((gridHeight, gridWidth), dtype = numba.complex64)

    delta0x = 1/width
    delta0y = 1/height

    if geometry == 'point':
        fac = math.pi*wavelength*depth

        for x in range(centreX + 1) :
             uSq = (delta0x*x)**2

             for y in range(centreY + 1) :
            
                 vSq = (delta0y*y)**2

                 phase = fac*(uSq + vSq)

                 # This is about as twice as fast as using np.exp(1j * phase)
                 propCorner.real[y,x] = math.cos(phase)
                 propCorner.imag[y,x] = math.sin(phase)
        
    elif geometry == 'plane':   
        fac = 2 * math.pi * depth / wavelength
        for x in range(centreX + 1) :
             alphaSq = (float(wavelength) * x * delta0x)**2

             for y in range(centreY + 1) :
                 betaSq = (float(wavelength) * y * delta0y)**2
                 if alphaSq + betaSq < 1:
                     phase = fac * np.sqrt(1 - alphaSq - betaSq) 
                     
                     # This is about as twice as fast as using np.exp(1j * phase)
                     propCorner.real[y,x] = math.cos(phase)
                     propCorner.imag[y,x] = math.sin(phase)
    else:
        raise Exception("Invalid geometry.")               
            
      
    # Duplicate the top left quadrant into the other three quadrants as
    # this is quicker then explicitly calculating the values
    prop[:centreY + 1, :centreX + 1] = propCorner                      # top left
    prop[:centreY + 1, centreX:] = (np.fliplr(propCorner[:, 1:]) )     # top right
    prop[centreY:, :] = np.flipud(prop[1:centreY + 1, :])              # bottom left

   
    return prop
