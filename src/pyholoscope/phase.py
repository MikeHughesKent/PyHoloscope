# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

@author: Mike Hughes, Applied Optics Group, University of Kent

This file contains general functions, mostly for handling and displaying phase
maps.

"""

import numpy as np
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
from pyholoscope.utils import extract_central

 

def relative_phase(img, background):
    """ Removes global phase from image using reference (background) field.
    
    The function works on both fields (complex arrays) and phase maps (real 
    arrays), and returns a corrected field/phase map of the same type.
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex. If real
                        it is taken to be phase map, otherwise if complex
                        it is the field.
          background :  ndarray
                        2D numpy array, real or complex. Backround phase/field
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
    arrays), and returns a corrected field/phase map of the same type.
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex. If real
                        it is taken to be phase map, otherwise if complex
                        it is the field.
    
    Keyword Arguments:
          roi        :  pyholoscope.Roi
                        region of interest to make phase relative to. In the output image
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
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real. Unwrapped phase.
          
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
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real. Wrapped phase.
    """
    
    img = unwrap_phase(img)

    return img



def synthetic_DIC(img, shearAngle = math.pi):
    """ Generates a simple, non-rigorous DIC-style image for display. 
    
    The ouput should appear similar to a relief map, with dark and light 
    regions correspnding to positive and negative phase gradients along the
    shear angle direction (default is horizontal = 0 rad). The input must
    be the complex field, not a phase map.
    
    Parameters:
          img        :  ndarray
                        2D numpy array, complex, the field.
    
    Keyword Arguments:
          shearAngle :  float
                        angle in radians of the shear direction. 
                        (default = pi)
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
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex, the field or phase map
    
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
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex, the field or phase map
    
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
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex, the field or phase map
     
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
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex, the field or phase map   
    """
    
    if np.iscomplexobj(img):
        return np.angle(np.sum(img))    # The way to average the phase in a field
    else:
        return np.mean(img)
    




   
   



        
