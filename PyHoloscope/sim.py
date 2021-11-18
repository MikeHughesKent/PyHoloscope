# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:20:10 2021

@author: AOG
"""

import math
import cmath
import numpy as np

class Tracker:
    def __init__(self):
        pass
    
    
    def __str__(self):
        return "Sim"
    
def test():
    print("hello world") 
        
        
def offAxis(objectField, wavelength, pixelSize, tiltAngle):
    
    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength      

    # Check size of object
    nPoints = np.shape(objectField)[0]

    # Generate tilted reference field
    refField = np.ones((nPoints, nPoints))
    refFreq = k * math.sin(tiltAngle)
    (xM, yM) = np.meshgrid(range(nPoints), range(nPoints))
    refField = refField * np.exp(1j * refFreq * pixelSize * (xM + yM)/np.sqrt(2))

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
    
    
    