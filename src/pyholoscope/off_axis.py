# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Mike Hughes, Applied Optics Group, University of Kent

PyHoloscope is a python library for holographic microscopy.

This file contains functions for handling off-axis holograms.


"""
import math
import numpy as np
try:
    import cupy as cp
    cudaAvailable = True
except:
    cudaAvailable = False
    

def off_axis_demod(cameraImage, cropCentre, cropRadius, **kwargs):
    """ Removes spatial modulation from off axis hologram. cropCentre is the location of
    the modulation frequency in the Fourier Domain, cropRadius is the size of
    the spatial frequency range to keep around the modulation frequency (in FFT pixels)    
    """
    
    returnFFT = kwargs.get('returnFFT', False)
    mask = kwargs.get('mask', None)
    cuda = kwargs.get('cuda', False)
    
    # Size of image in pixels (assume square);
    nPoints = np.min(np.shape(cameraImage))
    cameraImage = cameraImage[0:nPoints, 0:nPoints]       
     
    # Make a circular mask
    if mask is None:
        [xM, yM] = np.meshgrid(range(cropRadius * 2), range(cropRadius *2))
        mask = (xM - cropRadius)**2 + (yM - cropRadius)**2 < cropRadius**2
        mask = mask.astype('complex')
  
    # Apply 2D FFT
    if cuda is False or cudaAvailable is False:
        cameraFFT = np.fft.rfft2(cameraImage)
    else:
        cameraFFT = cp.fft.rfft2(cp.array(cameraImage))
   
    # Shift the ROI to the centre
    shiftedFFT = cameraFFT[round(cropCentre[1] - cropRadius): round(cropCentre[1] + cropRadius),round(cropCentre[0] - cropRadius): round(cropCentre[0] + cropRadius)]

    # Apply the mask
    if cuda is True and cudaAvailable is True:
        mask = cp.array(mask)
    maskedFFT = shiftedFFT * mask

    # Reconstruct complex field
    if cuda is False or cudaAvailable is False:
        reconField = np.fft.ifft2(np.fft.fftshift(shiftedFFT))
    else:
        reconField = cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(shiftedFFT)))
   
    if returnFFT:
        if cuda is True and cudaAvailable is True:
            try:
                cameraFFT = cp.asnumpy(cameraFFT)
            except:
                pass
        return reconField, np.log(np.abs(cameraFFT) + 0.000001)
    
    else:
        return reconField
    

def off_axis_find_mod(cameraImage, maskFraction = 0.1):
    """ Finds the location of the off-axis holography modulation peak in the FFT. Finds
    the peak in the positive x region.    
    """
    
    # Apply 2D FFT
    cameraFFT = np.transpose(np.abs(np.fft.rfft2(cameraImage)) ) 
    
 
    # Mask central region
    imSize = min(np.shape(cameraImage)[:1])
    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2    
    
    # Need to crop out DC otherwise we will find that. Set the areas around
    # dc (for both quadrants) to zero. The size of the masked area is maskFraction * the
    # size of the image (smallest dimension)
    maskSize = int(np.min(np.shape(cameraImage)) * maskFraction)

    cameraFFT[:maskSize,:maskSize] = 0
    cameraFFT[:maskSize:,-maskSize:] = 0
   
 
    peakLoc = np.unravel_index(cameraFFT.argmax(), cameraFFT.shape)
    
    return peakLoc


def off_axis_find_crop_radius(cameraImage):
    """ Estimates the correct off axis crop radius based on modulation peak position
    """
    
    h = np.shape(cameraImage)[0]
    w = np.shape(cameraImage)[1]
    cx = w / 2
    cy = h / 2

    peakLoc = off_axis_find_mod(cameraImage)
        
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
    """ Predicts the location of the modulation peak (i.e. carrer frequency) in the
    FFT. Returns the distance of the peak from the centre (dc) of the FFT in pixels.
    """
           
    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength     
     
    # Spatial frequency of mdulation
    refFreq = k * math.sin(tiltAngle)
    
    # Spatial frequency in camera pixels
    refFreqPx = refFreq / pixelSize
    
    # Pixel in Fourier Domain
    modFreqPx = 2 / refFreqPx
    
    return modFreqPx


def off_axis_predict_tilt_angle(cameraImage, wavelength, pixelSize):
    """ Predicts the reference beam tilt based on the modulation of the camera image
    and specified wavelength and pixel size.
    """    
    
    # Wavenumber
    k = 2 * math.pi / wavelength    

    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    
    # Find the location of the peak
    peakLoc = off_axis_find_mod(cameraImage)
    
    hPixelSF = 1 / (2 * pixelSize * np.shape(cameraImage)[0])
    vPixelSF = 1 / (2 * pixelSize * np.shape(cameraImage)[1])
    
    spatialFreq = np.sqrt( (hPixelSF * (peakLoc[0] - cx))**2  + (vPixelSF * (peakLoc[1] - cy) )**2)
   
    tiltAngle = math.asin(spatialFreq / k)
    
    return tiltAngle