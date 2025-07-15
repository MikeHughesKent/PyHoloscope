# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

This file contains functions for working with off-axis holograms.
"""

import math
import numpy as np
import scipy

import matplotlib.pyplot as plt

try:
    import cupy as cp
    cudaAvailable = True
except:
    cudaAvailable = False
    
from pyholoscope.utils import extract_central, dimensions
    

def off_axis_demod(hologram, cropCentre, cropRadius, returnFull = False, returnFFT = False, 
                   mask = None, cuda = False):
    """ Removes spatial modulation from off-axis hologram to obtain complex field.
    
    By default, returns the complex field as a 2D numpy array of size 
    2 * cropRadius. If returnFull is True, the returned
    array will instead by the same size as the input hologram. If returnFFT is
    True, function returns a tuple (field, FFT) where FFT is a log scaled
    image of the FFT of the hologram (2D numpy array, real).
        
    Arguments:
          hologram   : numpy.ndarray
                       2D numpy array, real, raw hologram
          cropCentre : tuple of (int, int). 
                       pixel location in FFT of modulation frequency
          cropRadius : int or (int, int)
                       semi-diameter of sqaure or rectangle to extract
                       around modulation frequency. Provide a single int
                       for a sqaure area or a tuple of (w,h) for a rectangle.
        
    Keyword Arguments:
          returnFull : boolean
                       if True, the returned reconstruction will be the same
                       size as the input hologram, otherwise it will be
                       2 * cropRadius. (Default is False)
          returnFFT  : boolean 
                       if True will return a tuple of (demod image,
                       log scaled FFT) for display purposes
                       (Default is False)
          mask       : ndarray
                       2D complex array. Custom mask to use around
                       demodulation frequency. Must match size of 
                       (cropRadiusX, cropRadiusY)
          cuda      :  boolean
                       if True GPU will be used if available.  
    Returns:       
          numpy. ndarray   : reconstructed field as complex numpy array or 
                             tuple of (ndarray, ndarray) if returnFFT is True       
    """
           
    # Size of image in pixels 
    height, width = np.shape(hologram)    
    
    cropCentre = dimensions(cropCentre)
    cropRadius = dimensions(cropRadius)
   
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
    shiftedFFT = cameraFFT[round(cropCentre[1] - cropRadius[1]): round(cropCentre[1] + cropRadius[1]),round(cropCentre[0] - cropRadius[0]): round(cropCentre[0] + cropRadius[0])]

    # Apply the mask
    if mask is not None:
        assert np.shape(mask) == np.shape(shiftedFFT), "Incorrect mask size."
        maskedFFT = shiftedFFT * mask
    else:
        maskedFFT = shiftedFFT        
 
    if returnFull:
        h,w = np.shape(hologram)
        h2, w2, = np.shape(maskedFFT)
        x,y = round((w - w2) / 2), round((h - h2)/2)
        output = np.zeros((h,w), dtype = 'complex')
        output[y: y + h2, x: x + w2] = maskedFFT
        maskedFFT = output
         

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
    Returns:
          tuple of (int, int), modulation location in FFT (y location, x location)                     
    """
    
    # Apply 2D FFT
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
    """ Estimates the off-axis crop radius based on modulation peak position. If the
    hologram is square, this is the radius of a circle, otherwise if it is rectangular 
    than the crop radius is a tuple of (y radius, x radius), corresponding to 
    half the lengths of the two axes of an ellipse.
    
    Arguments:
          hologram     : numpy.ndarray
                         raw hologram, 2D real array
         
    Keyword Arguments:
          maskFraction : float 
                         between 0 and 1, fraction of image around d.c. to 
                         mask to avoid the d.c. peak being detected 
                         (default = 0.1).                         
    Returns:
          tuple of (int, int) = (y radius, x radius)                     
    """
    
    h = np.shape(hologram)[0]
    w = np.shape(hologram)[1]
 
    peakLocY, peakLocX = off_axis_find_mod(hologram, maskFraction = maskFraction)
    
    # The crop radii will have the same ratio as the width and height of the hologram
    aspectRatio = h/w
    
    peakLocSquare = (peakLocY * aspectRatio, peakLocX)    
    
    # In the optimal case, the radius is 1/3rd of the modulation position
    if peakLocX < h /2:
        cropRadiusSquare = np.sqrt(peakLocSquare[0]**2 + peakLocSquare[1]**2) / 3
        cropRadiusSquare = min(cropRadiusSquare, peakLocY, int(w - peakLocY), peakLocX, int(h/2 - peakLocX) )
    else:
        cropRadiusSquare = np.sqrt(peakLocSquare[0]**2 + (w*2 - peakLocSquare[1])**2) / 3
        cropRadiusSquare = min(cropRadiusSquare, peakLocY, int(w - peakLocY), peakLocX - h/2 * aspectRatio, int(h * aspectRatio - peakLocX) )
      
    cropRadiusX = int(round(cropRadiusSquare))
    cropRadiusY = int(round(cropRadiusSquare / aspectRatio))
   
    return cropRadiusY, cropRadiusX


def off_axis_predict_mod(wavelength, pixelSize, numPixels, tiltAngle, rotation = 0): 
    """ Predicts the location of the modulation peak in the FFT.
   
    Arguments:
          wavelegnth   : float
                         light wavelength in metres
          pixelSize    : float
                         hologram physical pixel size in metres
          numPixels    : int or (int, int)
                         hologram size in pixels,             
          tiltAngle    : float
                         angle of reference beam on camera in radians 
                         
    Keyword Arguments:
          rotation     : float
                         rotation of tilt with respect to x axis, in radians (default is 0)                     
    
    Returns:
          tuple of (int, int), location of modulation (x pixel, y pixel)
    
    """   
     
    # Spatial frequency of modulation
    refFreq = math.sin(tiltAngle) / wavelength
   
    # Spatial frequency at edge of FFT
    maxSF = 1 / (pixelSize * 2)
    
    imSizeX, imSizeY = dimensions(numPixels)
    
    # Pixel corresponding to frequency in Fourier Domain
    if rotation%math.pi < math.pi/2:
        modFreqPxX = round(refFreq / maxSF * np.abs(np.cos(rotation)) * imSizeX / 2)
        modFreqPxY = round(refFreq / maxSF * np.abs(np.sin(rotation)) * imSizeY / 2)
    else:
        modFreqPxX = round(refFreq / maxSF * np.abs(np.cos(math.pi  - rotation)) * imSizeX / 2)
        modFreqPxY = imSizeY - round(refFreq / maxSF * np.abs(np.sin(rotation)) * imSizeY / 2)        
    
    if modFreqPxX < 0: modFreqPxX = modFreqPxX + imSizeX
    if modFreqPxY < 0: modFreqPxY = modFreqPxY + imSizeY
    
    return modFreqPxX, modFreqPxY


def off_axis_predict_mod_distance(wavelength, pixelSize, numPixels, tiltAngle, rotation = 0): 
    """ Predicts the absolute distance of the modulation peak in the FFT from the dc.
   
    Arguments:
          wavelegnth   : float
                         light wavelength in metres
          pixelSize    : float
                         hologram physical pixel size in metres
          numPixels    : int or (int, int)
                         hologram size in pixels,             
          tiltAngle    : float
                         angle of reference beam on camera in radians 
                         
    Keyword Arguments:
          rotation     : float
                         rotation of tilt with respect to x axis, in radians (default is 0)                     
    
    Returns:
          float        : distance in pixels
    
    """
   
    x,y = off_axis_predict_mod(wavelength, pixelSize, numPixels, tiltAngle, rotation)
    
    return math.sqrt(x**2 + y**2)


def off_axis_predict_tilt_angle(hologram, wavelength, pixelSize, maskFraction = 0.1):
    """ Returns the reference beam tilt based on the hologram modulation. The angle
    is returned in radians.
    
    Arguments:
          hologram     : ndarray
                         2D numpy array, real, hologram
          wavelength   : float
                         light wavelength in metres
          pixelSize    : float
                         hologram physical pixel size in metres
    
    Optional Keyword Arguments:
          maskFraction : float 
                         between 0 and 1, fraction of image around d.c. to 
                         mask to avoid the d.c. peak being detected 
                         (default = 0.1).  
                         
    Returns:
          float        : tilt angle in radians
    """
    
    # Wavenumber
    k = 2 * math.pi / wavelength    

    h, w = np.shape(hologram)[:2]
    
    # Find the location of the peak
    peakLoc = off_axis_find_mod(hologram, maskFraction = maskFraction)

    # Pixel sizes in FFT (the spatial frequency)
    vPixelSF = 1 / (pixelSize * np.shape(hologram)[0])
    hPixelSF = 1 / (pixelSize * np.shape(hologram)[1])
    
    # Depending on quadrant could be relative to either top-left or
    # top-right corner, so check both and use the closest distance    
    peakDist1 = np.sqrt((vPixelSF * peakLoc[1])**2 + (hPixelSF * peakLoc[0])**2)
    peakDist2 = np.sqrt((vPixelSF * peakLoc[1])**2 + (hPixelSF * (peakLoc[0] - w))**2)
    spatialFreq = min(peakDist1, peakDist2) 
 
    tiltAngle = math.asin(2 * math.pi * spatialFreq / k)
    
    return tiltAngle