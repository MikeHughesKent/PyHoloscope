# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Mike Hughes, Applied Optics Group, University of Kent

PyHoloscope is a python library for holographic microscopy.

This file contains general functions.

"""

import numpy as np
from matplotlib import pyplot as plt
import math
import scipy
import scipy.optimize
import warnings
import time
try:
    import cupy as cp
except:
    pass
    
from PIL import Image
import cv2 as cv

from pyholoscope.roi import Roi
from pyholoscope.focus_stack import FocusStack
from pyholoscope.prop_lut import PropLUT

from skimage.restoration import unwrap_phase

def __init__():
    pass


def pre_process(img, background = None, normalise = None, window = None, downsample = 1.):
    """ Carries out steps required prior to refocus - background correction and 
    windowing. Also coverts image to either float32 (if input img is real) or
    complex64 (if input img is complex). Finally, image is cropped to a square
    as non-square images are not currently supported.
    Required Parameters:
          img       :  raw hologram, 2D numpy array, real or complex
        
    Optional Parameters:
          background : backround hologram to be subtracted, 2D numpy array (real)
          normalise  : background hologram to be divided, 2D numpy array (real)
          window     : window to smooth edges, 2D numpy array (real). Will be resized if necessary.
          downsample : factor to downsample image by
    """    
    
    # We will make it float, or a complex float
      
      
    if np.iscomplexobj(img):
        imType = 'complex128'
    else:
        imType = 'float32'     
       
    # Background subtraction  
    if background is not None:
       
        if np.iscomplexobj(img):
            imgAmp = np.abs(img)
            imgPhase = np.angle(img)
            imgOut = np.zeros_like(img).astype(imType)
            imgOut.real = (imgAmp - background) * np.cos(imgPhase)
            imgOut.imag = (imgAmp - background) * np.sin(imgPhase)
        else:
            imgOut = img - background
    else:
        imgOut = img.astype(imType)
    
    
    # Background normalisation 
    if normalise is not None:
        imgOut = imgOut / normalise 
       
    
    # Apply downsampling
    if downsample != 1:                
        imgOut = cv.resize(imgOut, (int(np.shape(img)[1]/ downsample), int(np.shape(img)[0] / downsample) )   )
    
    # Ensure it is square
    minSize = np.min(np.shape(imgOut))
    imgOut = imgOut[:minSize, :minSize]
     
    # Apply window
    if window is not None:
        
        # If the window is the wrong size, reshape it to match hologram
        if np.shape(window) != np.shape(imgOut):
            warnings.warn('Window needed resizing, may effect processing speed.')
            window = cv.resize(window, (np.shape(imgOut)[1],np.shape(imgOut)[0])  )

        if np.iscomplexobj(img):
            imgOut.imag = imgOut.imag * window
            imgOut.real = imgOut.real * window
            imgOut[imgOut == -0+0j] = 0j     # Otherwise phase angle looks weird when plotted
        else:
            imgOut = imgOut * window 
            
    return imgOut

 

def relative_phase(img, background):
    """ Remove global phase from image using reference (background) field 
    if img/background are real, they are both taken to be phases
    if img/background are complex, they are both taken to be complex fields
    """    
    
    
    # If both inputs are phases, simply subtract        
    if not np.iscomplexobj(img) and not np.iscomplexobj(background):
        return img - background
    
    # If both input are fields, divide
    if np.iscomplexobj(img) and np.iscomplexobj(background):
        return img / background * np.abs(background)
    
   
        

def stable_phase(img, roi = None):
    """ Subtracts the mean phase from the phase map, removing global phase
    fluctuations. Can accept complex img, as a field, or a real img, which
    is unwrapped phase in radians 
    """
   
    if roi is not None:
        imgCrop = roi.crop(img)
    else:
        imgCrop = img 
   
    if np.iscomplexobj(img):
        phase = np.angle(img)
        phaseCrop = np.angle(imgCrop)
    else:
        phase = img
        phaseCrop = imgCrop   
    
    avPhase = mean_phase(imgCrop)

    phaseOut = phase - avPhase

    if np.iscomplexobj(img):             
        return np.abs(img) * np.exp(1j * phaseOut)
    else:     
        return phaseOut


def obtain_tilt(img):
    """ Estimates the global tilt in the 2D unwrapped phase (e.g. caused by tilt in coverglass). img
    should be unwrapped phase (real)
    """
    
    tiltX, tiltY = np.gradient(img)
    tiltX = np.mean(tiltX)
    tiltY = np.mean(tiltY)
    
    mx, my = np.indices(np.shape(img))
    
    tilt = mx * tiltX + my * tiltY
   
    return tilt
 
    
def phase_unwrap(img):
    """ 2D phase unwrapping. img should be wrapped phase (real)
    """    
    img = unwrap_phase(img)

    return img


def fourier_plane_display(img):
    """ Return a log-scale Fourier plane for display
    """
    cameraFFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    return cameraFFT    


def synthetic_DIC(img, **kwargs):

    """ Generates a simple, non-rigorous DIC-style image for display. The image
    should appear similar to a relief map, with dark and light regions
    correspnding to positive and negative phase gradients along the
    shear angle direction (default is horizontal = 0 rad). Phase gradient
    is multiplied by the image intensity. 'img' should be a complex numpy array.    
    """
    
    shearAngle = kwargs.get('shearAngle', 0)
    
    # Calculate gradient on original image and image phase shifted by pi. Using
    # the smallest phase gradient avoids effects due to phase wrapping
    sobelC1 = phase_gradient_amp(img)
    sobelC2 = phase_gradient_amp(img * np.exp(1j * math.pi))
    
    use1 = np.abs(sobelC1) < np.abs(sobelC2)
    
    sobelC1[np.invert(use1)] = 0
    sobelC2[use1] = 0
    sobelC = sobelC1 + sobelC2
    # Rotate the gradient to shear angle
    sobelC = sobelC * np.exp(1j * shearAngle)
       
    # DIC is product of phase gradient along one direction and image intensity
    DIC = np.real(sobelC) * (np.max(np.abs(img)) - np.abs(img)) 
    # Not sure how best to involvw amplitude here
    # DIC = np.real(sobelC) * (-np.abs(img))
        
    return DIC


def phase_gradient_amp(img):
    """ Returns the ampitude of the phase gradient
    """
    
    # Phase gradient in x and y directions
    sobelx = cv.Sobel(np.angle(img),cv.CV_64F,1,0)                  # Find x and y gradients
    sobely = cv.Sobel(np.angle(img),cv.CV_64F,0,1)
    sobelC = sobelx + 1j * sobely

    return sobelC


def phase_gradient(img):
    """ Produces a phase gradient (magnitude) image. img should be a complex numpy
    array
    """
    
    # Phase gradient in x and y directions
    phaseGrad1 = np.abs(phase_gradient_amp(img))
    phaseGrad2 = np.abs(phase_gradient_amp(img * np.exp(1j * math.pi)))
   
    phaseGrad = np.minimum(phaseGrad1, phaseGrad2)
    
    return phaseGrad


def mean_phase(img):
    """Returns the mean phase in a complex field
    """
    if np.iscomplexobj(img):
        meanPhase = np.angle(np.sum(img))
    else:
        meanPhase = np.mean(img)
    return meanPhase


def relative_phase_ROI(img, roi):    
    """ Makes the phase in an image relative to the mean phase in specified ROI
    """
    avPhase = mean_phase(roi.crop(img))
    outImage = img / np.exp(1j * avPhase)
    
    return outImage


   
   



        
