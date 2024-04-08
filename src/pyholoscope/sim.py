# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

The Sim package provide utilites for simulating holograms.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import math
import cmath
import numpy as np
from pyholoscope import dimensions
        
def off_axis(objectField, wavelength, pixelSize, tiltAngle, rotation = math.pi / 4):
    """ Generates simulated off-axis hologram.
    
    Arguments:
        objectField : ndarray, complex
                      complex field representing object
        wavelength  : float
                      wavelength of simulated light source
        pixelSize   : float
                      real size of pixels in hologram
        tiltAngle   : float
                      angle of reference beam on camera in radians
  
    Keyword Arguments:
        rotation    : float
                      rotation of the tilt with respect to x axis in rad 
                      (Default is pi/4 = 45 deg)
           
    """
    
    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength      

    # Check size of object
    nPointsY, nPointsX = np.shape(objectField)

    # Generate tilted reference field
    refField = np.ones((nPointsY, nPointsX))
    refFreq = k * math.sin(tiltAngle)
    (xM, yM) = np.meshgrid(range(nPointsX), range(nPointsY))


    refField = refField * np.exp(1j * (refFreq * pixelSize * (xM * np.cos(rotation) + yM * np.sin(rotation))))
    

    # Field at camera is superposition of images of object and reference. 
    # We are assuming 1:1 magnification here.
    cameraField = refField + objectField;

    # Intensity at camera is square of field, 
    cameraIntensity = np.abs(cameraField)**2
    cameraIntensity = (cameraIntensity / np.max(cameraIntensity) * 255)
    
    # Add noise to camera image and sample at 8 bits
    # This is Gaussian, should really by changed to Poisson but okay
    # providing noise is not too large. Any negative values from
    #% Gaussian noise are rounded to zero by uint8
    #cameraIntensity = uint8(cameraIntensity + randn(size(cameraIntensity)) .* noise);
    
    return cameraIntensity