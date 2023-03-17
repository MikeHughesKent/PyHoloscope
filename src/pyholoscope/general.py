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


def pre_process(img, **kwargs):
    """ Carries out steps required prior to refocus - background correction and 
    windowing. Also coverts image to either float32 (if input img is real) or
    complex64 (if input img is complex). Finally, image is cropped to a square
    as non-square images are not currently supported.
    """    

    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    
    if np.iscomplex(img[0,0]):
        imType = 'complex64'
    else:
        imType = 'float32'
                
    
    if background is not None:
        print("back sub")
        imgOut = img.astype(imType) - background.astype(imType)
    else:
        imgOut  = img.astype(imType)
        
    minSize = np.min(np.shape(imgOut))
    imgOut = imgOut[:minSize, :minSize]
            
    if window is not None:
        if np.iscomplex(img[0,0]):
            imgOut = np.abs(imgOut) * window * np.exp(1j * np.angle(imgOut) * window)
        else:
            imgOut = imgOut * window.astype(imType)
            
    return imgOut


def post_process(img, **kwargs):
    """ Processing after refocus - background subtraction and windowing"""

    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    
    if np.iscomplex(img[0,0]):
        imType = 'complex64'
    else:
        imType = 'float32'
    
    if background is not None:
        imgOut = img.astype(imType) - background.astype(imType)
    else:
        imgOut  = img.astype(imType)
            
    if window is not None:
        imgOut = imgOut * window
            
    return imgOut
 

def relative_phase(img, background):
    """ Remove global phase from complex image using reference (background) field 
    """    
    
    if np.iscomplexobj(img):
        phase = np.angle(img)
    else:
        phase = img
        
    if np.iscomplexobj(background):
        backgroundPhase = np.angle(background)
    else:
        backgroundPhase = background
        
    phaseOut = phase - backgroundPhase    
    
    if np.iscomplexobj(img):             
        return np.abs(img) * np.exp(1j * phaseOut)
    else:     
        return phaseOut
        

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


   
   



        
