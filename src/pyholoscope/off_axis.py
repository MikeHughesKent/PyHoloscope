# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

@author: Mike Hughes, Applied Optics Group, University of Kent

This file contains functions for working with off-axis holograms.

"""

import math
import numpy as np
import scipy

try:
    import cupy as cp
    cudaAvailable = True
except:
    cudaAvailable = False
    
from pyholoscope.utils import extract_central
    

def off_axis_demod(hologram, cropCentre, cropRadius, returnFFT = False, 
                   mask = None, cuda = False):
    """ Removes spatial modulation from off-axis hologram to obtain field.
    
    By default, returns the complex field as a 2D numpy array. The size of
    the returned array is (cropRadius*2, cropRadius*2). If returnFFT is
    True, function returns a tuple (field, FFT) where FFT is a log scaled
    image of the FFT of the hologram (2D numpy array, real).
        
    Arguments:
          hologram   : ndarray
                       2D numpy array, real, raw hologram
          cropCentre : tuple of (int, int). 
                       pixel location in FFT of modulation frequency
          cropRadius : int
                       radius of circle around modulation frequency to use
        
    Keyword Arguments:
          returnFFT  : boolean 
                       if True will return a tuple of (demod image,
                       log scaled FFT) for display purposes
                       subtracted (default = None)
          mask       : ndarray
                       2D numpy array, complex. Custom mask to use around
                       demodulation frequency instead of default circle. Must
                       be sqaure of size (cropRadius*2, cropRadius*2).
          cuda      :  boolean
                       if True GPU will be used if available.  
    """
    
    
    hologram = extract_central(hologram)
    
    # Size of image in pixels (assume square)
    nPoints = np.min(np.shape(hologram))
 
     
    # Make a circular mask
    if mask is None:
        [xM, yM] = np.meshgrid(range(cropRadius * 2), range(cropRadius *2))
        mask = (xM - cropRadius)**2 + (yM - cropRadius)**2 < cropRadius**2
        mask = mask.astype('complex')
  
    # Apply 2D FFT
    if cuda is False or cudaAvailable is False:
        cameraFFT = scipy.fft.rfft2(hologram)
    else:
        if type(hologram) is np.ndarray:
            hologram = cp.array(hologram)
        if type(mask) is np.ndarray:
            mask = cp.array(mask)
        cameraFFT = cp.fft.rfft2(cp.array(hologram))
   
    # Shift the ROI to the centre
    shiftedFFT = cameraFFT[round(cropCentre[1] - cropRadius): round(cropCentre[1] + cropRadius),round(cropCentre[0] - cropRadius): round(cropCentre[0] + cropRadius)]

    # Apply the mask
    maskedFFT = shiftedFFT * mask

    # Reconstruct complex field
    if cuda is False or cudaAvailable is False:
        reconField = scipy.fft.ifft2(scipy.fft.fftshift(maskedFFT))
    else:
        reconField = cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(maskedFFT)))
   
    if returnFFT:
        if cuda is True and cudaAvailable is True:
            try:
                cameraFFT = cp.asnumpy(cameraFFT)
            except:
                pass
        return reconField, np.log(np.abs(cameraFFT) + 0.000001) # Stops log(0)
    
    else:
        return reconField
    

def off_axis_find_mod(hologram, maskFraction = 0.1):
    """ Finds the location of the off-axis holography modulation peak in the FFT. 
    
    Arguments:
          hologram     : ndarray
                         2D numpy array, real, raw hologram
         
    Keyword Arguments:
          maskFraction : float 
                         between 0 and 1, fraction of image around d.c. to 
                         mask to avoid the d.c. peak being detected 
                         (default = 0.1).
    """
    
    # Apply 2D FFT
    hologram = extract_central(hologram)
    cameraFFT = np.transpose(np.abs(scipy.fft.rfft2((hologram)))) 
    
    # Need to crop out DC otherwise we will find that. Set the areas around
    # dc (for both quadrants) to zero. The size of the masked area is maskFraction * the
    # size of the image (smallest dimension)
    maskSize = int(np.min(np.shape(hologram)) * maskFraction)
    cameraFFT[:maskSize,:maskSize] = 0
    cameraFFT[:maskSize:,-maskSize:] = 0
 
    peakLoc = np.unravel_index(cameraFFT.argmax(), cameraFFT.shape)
    
    return peakLoc


def off_axis_find_crop_radius(hologram, maskFraction = 0.1):
    """ Estimates the off axis crop radius based on modulation peak position.
    
    Arguments:
          hologram     : ndarray
                         2D numpy array, real, raw hologram
         
    Keyword Arguments:
          maskFraction : float 
                         between 0 and 1, fraction of image around d.c. to 
                         mask to avoid the d.c. peak being detected 
                         (default = 0.1).
    """
    
    h = np.shape(hologram)[0]
    w = np.shape(hologram)[1]
 
    peakLoc = off_axis_find_mod(hologram, maskFraction = maskFraction)
        
    # Depending on quadrant could be relative to either top-left or
    # top-right corner, so check both and use the closest distance
    peakDist1 = np.sqrt((peakLoc[0])**2 + (peakLoc[1])**2)
    peakDist2 = np.sqrt((peakLoc[0])**2 + (peakLoc[1] - w)**2)
    peakDist = min(peakDist1, peakDist2)
    
    # In the optimal case, the radius is 1/3rd of the modulation position
    cropRadius = math.floor(peakDist / 3)
    
    # Ensure it doesn't run off edge of image
    cropRadius = min (cropRadius, peakLoc[0], int(h / 2 )- peakLoc[0], peakLoc[1], w - peakLoc[1] )
    
    return cropRadius


def off_axis_predict_mod(wavelength, pixelSize, tiltAngle): 
    """ Predicts the location of the modulation peak in the FFT. 
    
    Returns the distance of the peak from the dc of the FFT in pixels.
    
    Arguments:
          wavelegnth   : float
                         light wavelength in metres
          pixelSizze   : float
                         hologram physical pixel size in metres
          tiltAngle    : float
                         angle of reference beam on camera in radians    
    """
           
    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength     
     
    # Spatial frequency of modulation
    refFreq = k * math.sin(tiltAngle)
    
    # Spatial frequency in camera pixels
    refFreqPx = refFreq / pixelSize
    
    # Pixel in Fourier Domain
    modFreqPx = 2 / refFreqPx
    
    return modFreqPx


def off_axis_predict_tilt_angle(hologram, wavelength, pixelSize, maskFraction = 0.1):
    """ Returns the reference beam tilt based on the hologram modulation.
    
    Returns the angle in radians.
    
    Arguments:
          hologram     : ndarray
                         2D numpy array, real, hologram
          wavelength   : float
                         light wavelength in metres
          pixelSizze   : float
                         hologram physical pixel size
    
    Optional Keyword Arguments:
          maskFraction : float 
                         between 0 and 1, fraction of image around d.c. to 
                         mask to avoid the d.c. peak being detected 
                         (default = 0.1).        
    """
    
    # Wavenumber
    k = 2 * math.pi / wavelength    

    h, w = np.shape(hologram)[:2]
    
    # Find the location of the peak
    peakLoc = off_axis_find_mod(hologram, maskFraction = maskFraction)

    # Pixel sizes in FFT (the spatial frequency)
    vPixelSF = 1 / (2 * pixelSize * np.shape(hologram)[0])
    hPixelSF = 1 / (2 * pixelSize * np.shape(hologram)[1])

    
    # Depending on quadrant could be relative to either top-left or
    # top-right corner, so check both and use the closest distance    
    peakDist1 = np.sqrt((vPixelSF * peakLoc[0])**2 + (hPixelSF * peakLoc[1])**2)
    peakDist2 = np.sqrt((vPixelSF * peakLoc[0])**2 + (hPixelSF * peakLoc[1] - w)**2)
    spatialFreq = min(peakDist1, peakDist2) 
 
    tiltAngle = math.asin(spatialFreq / k)
    
    return tiltAngle