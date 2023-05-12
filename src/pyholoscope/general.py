# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

@author: Mike Hughes, Applied Optics Group, University of Kent

This file contains general functions, mostly for handling and displaying phase
maps.

"""

import numpy as np
from matplotlib import pyplot as plt
import math
import scipy
import scipy.optimize
import warnings
import time

from skimage.restoration import unwrap_phase
    
from PIL import Image

import cv2 as cv

from pyholoscope.roi import Roi
from pyholoscope.focus_stack import FocusStack
from pyholoscope.prop_lut import PropLUT


def pre_process(img, background = None, normalise = None, window = None, downsample = 1., numba = False):
    
    """ Carries out steps required prior to refocusing.
    
    This includes background correction, normalisation and windowing. It also 
    coverts image to either float64 (if input img is real) or
    complex128 (if input img is complex). Finally, the image is cropped to a 
    square and, if requested, downsampled.
    
    Arguments:
          img       :  raw hologram, 2D numpy array, real or complex
        
    Optional Keyword Arguments:
          background : 2D numpy array (real), backround hologram to be 
                       subtracted (default = None)
          normalise  : 2D numpy array (real), background hologram to be 
                       divided (default = None)
          window     : 2D numpy array (real), window to smooth edges. Will be  
                       resized to match size of img if necessary. 
                       (default = None)
          downsample : float, factor to downsample image by (default = 1)
    """    
    
    # Ensure the input hologram is a float or a complex float          
    if np.iscomplexobj(img):
        imType = 'complex128'
    else:
        imType = 'float64'    
    
    if img.dtype != imType:
        img = img.astype(imType)  
        
    if background is not None:
        if background.dtype != imType:
            background = background.astype(imType)  

    if normalise is not None:
        if normalise.dtype != imType:
            normalise = normalise.astype(imType)      
       
    # Background subtraction  
    if background is not None:       
        if np.iscomplexobj(img):
            imgAmp = np.abs(img)
            imgPhase = np.angle(img)
            img.real = (imgAmp - background) * np.cos(imgPhase)
            img.imag = (imgAmp - background) * np.sin(imgPhase)
        else:
            img = img - background
    
    # Background normalisation 
    if normalise is not None:
        img = img / normalise        
    
    # Apply downsampling
    if downsample != 1:                
        img = cv.resize(img, (int(np.shape(img)[1]/ downsample), int(np.shape(img)[0] / downsample) )   )
    
    # Ensure it is square
    minSize = np.min(np.shape(img))
    img = img[:minSize, :minSize]
     
    # Apply window
    if window is not None:
        
        # If the window is the wrong size, reshape it to match hologram
        if np.shape(window) != np.shape(img):
            warnings.warn('Window needed resizing, may effect processing speed.')
            window = cv.resize(window, (np.shape(img)[1],np.shape(img)[0])  )

        if np.iscomplexobj(img):
            img.imag = img.imag * window
            img.real = img.real * window
            img[img == -0+0j] = 0j     # Otherwise phase angle looks weird when plotted
        else:
            img = img * window 
            
    return img

 

def relative_phase(img, background):
    """ Removes global phase from image using reference (background) field.
    
    The function works on both fields (complex arrays) and phase maps (real 
    arrays, and returns a corrected field/phase map of the same type.
    
    Required Arguments:
          img        :  2D numpy array, real or complex. If real
                        it is taken to be phase map, otherwise if complex
                        it is the field.
          background :  2D numpy array, real or complex. Backround phase/field
                        to subtract. Must be same type and size as img.          
    """    
    
    assert np.iscomplexobj(img) == np.iscomplexobj(background), "img and background must both be real or both be complex"
    assert np.shape(img) == np.shape(background), "img and background must be the same size."
    
    # If both inputs are phases, simply subtract        
    if not np.iscomplexobj(img) and not np.iscomplexobj(background):
        return img - background
    
    # If both input are fields, divide
    if np.iscomplexobj(img) and np.iscomplexobj(background):
        return img / background * np.abs(background)
    
 

def relative_phase_self(img, roi = None):    
    """ Makes the phase in an image relative to the mean phase in either
    the whole image or a specified ROI of the image.
    
    The function works on both fields (complex arrays) and phase maps (real 
    arrays, and returns a corrected field/phase map of the same type.
    
    Required Arguments:
          img        :  2D numpy array, real or complex. If real
                        it is taken to be phase map, otherwise if complex
                        it is the field.
    
    Optional Keyword Arguments:
          roi        : instance of pyholoscope.Roi, region of interest 
                       to make phase relative to. In the output image
                       the mean phase in this region will be zero. 
                       (default = None)
    """

    if roi is None:
        avPhase = mean_phase(img)
    else:    
        avPhase = mean_phase(roi.crop(img))
        
    if np.iscomplexobj(img):
        outImage = img * np.exp(1j * -1 * avPhase)
    else:
        outImage = img - avPhase 
        
    return outImage     
    
    
def obtain_tilt(img):
    """ Estimates the global tilt in the 2D unwrapped phase.
    
    This can be used to correcte tilts in the phase due to, for example, a 
    tilt in the coverglass. Returns a phase map approximating the tilt as 
    a 2D real numpy array.
    
    Required Arguments:
          img        :  2D numpy array, real. Unwrapped phase.
          
    """
    
    # If there is a tilt then there will be a phase gradient across the image
    tiltX, tiltY = np.gradient(img)
    tiltX = np.mean(tiltX)
    tiltY = np.mean(tiltY)
    
    mx, my = np.indices(np.shape(img))
    
    tilt = mx * tiltX + my * tiltY
   
    return tilt
 
    
def phase_unwrap(img):
    """ Returns unwrapped version of 2D phase map. 
    
    Required Arguments:
          img        :  2D numpy array, real. Wrapped phase.
    """
    
    img = unwrap_phase(img)

    return img


def fourier_plane_display(img):
    """ Returns a log-scale Fourier plane for display purposes.
    
    Returns a 2D numpy array, real.
    
    Required Arguments:
          img        :  2D numpy array, real or complex.
    """
    
    cameraFFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    
    return cameraFFT    


def synthetic_DIC(img, shearAngle = math.pi):
    """ Generates a simple, non-rigorous DIC-style image for display. 
    
    The ouput should appear similar to a relief map, with dark and light 
    regions correspnding to positive and negative phase gradients along the
    shear angle direction (default is horizontal = 0 rad). The input must
    be the complex field, not a phase map.
    
    Required Arguments:
          img        :  2D numpy array, complex, the field.
    
    Optional Keyword Arguments:
          shearAngle : float, angle in radians of the shear direction. 
          (default = 0)
    """
    
    # Calculate gradient on original image and image phase shifted by pi. Using
    # the smallest phase gradient avoids effects due to phase wrapping
    sobelC1 = phase_gradient_dir(img)
    sobelC2 = phase_gradient_dir(img * np.exp(1j * math.pi))
    
    use1 = np.abs(sobelC1) < np.abs(sobelC2)
      
    sobelC1[np.invert(use1)] = 0
    sobelC2[use1] = 0
    sobelC = sobelC1 + sobelC2
        
    # Rotate the gradient to shear angle
    sobelC = sobelC * np.exp(1j * shearAngle)
    
    # DIC is product of phase gradient along one direction and image intensity
    DIC = np.real(sobelC) * (np.max(np.abs(img)) - np.abs(img)) 
    
    return DIC


def phase_gradient_dir(img):
    """ Returns the directional phase gradient of an image.
    
    The output is a complex 2D numpy array, with the horizontal and vertical
    gradients encoded in the real and imaginary parts.
    
    Required Arguments:
          img        :  2D numpy array, real or complex, the field or phase map
    
    """
    
    if np.iscomplexobj(img):
        img = np.angle(img)
    
    # Phase gradient in x and y directions
    sobelx = cv.Sobel(img,cv.CV_64F,1,0)     
    sobely = cv.Sobel(img,cv.CV_64F,0,1)
    
    # Encode in complex array
    sobelC = sobelx + 1j * sobely

    return sobelC


def phase_gradient_amp(img):
    """ Returns the amplitude of the phase gradient of an image.
    
    This isn't very useful by itself if applied to wrapped phase as it
    find an edge whenever the phase is wrapped, phase_gradient avoids this 
    problem.
    
    Required Arguments:
          img        :  2D numpy array, real or complex, the field or phase map
    
    """
    
    # If we are given the field, calculate the phase map
    if np.iscomplexobj(img):
        img = np.angle(img)
        
    # Phase gradient in x and y directions
    sobelx = cv.Sobel(img,cv.CV_64F,1,0)                  # Find x and y gradients
    sobely = cv.Sobel(img,cv.CV_64F,0,1)

    # Return amplitude
    return np.sqrt(sobelx**2 + sobely**2)
    


def phase_gradient(img):
    """ Returns the amplitude of the phase gradient of an image.

    This function is able to handle wrapped phase without finding an edge
    at the wrap points. Returns a 2D numpy array, real containing the amplitude
    of the gradient at each point.
    
    Required Arguments:
          img        :  2D numpy array, real or complex, the field or phase map
     
    """
    
    # We calculate the phase gradient twice, once adding a pi phase shift. If
    # the phase has wrapped at a point, this will produce a much bigger phase
    # gradient which we remove by taking the minimum of the two gradients.
        
    phaseGrad1 = phase_gradient_amp(img)
    
    if np.iscomplexobj(img):
        phaseGrad2 = phase_gradient_amp(img * np.exp(1j * math.pi))
    else:
        phaseGrad2 = phase_gradient_amp(np.exp(1j * img))
    
    return np.minimum(phaseGrad1, phaseGrad2)
    


def mean_phase(img):
    """ Returns the mean phase of a field or phase map.
    
    Required Arguments:
          img        :  2D numpy array, real or complex, the field or phase map   
    """
    
    if np.iscomplexobj(img):
        return np.angle(np.sum(img))    # The way to average the phase in a field
    else:
        return np.mean(img)
    




   
   



        
