# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

@author: Mike Hughes, Applied Optics Group, University of Kent

This file contains general functions, mostly for handling and displaying phase
maps.

"""

import math
import warnings

import numpy as np
import scipy
import cv2 as cv


def pre_process(img, background = None, normalise = None, window = None, downsample = 1., numba = False, precision = 'single'):
    """ Carries out steps required prior to refocusing.  
    This includes background correction, normalisation and windowing and 
    downsampling It also coverts image to either to a float64 (if input img is real) or
    complex float (if input img is complex). 
    
    Parameters:
          img        :  ndarray
                        raw hologram, 2D numpy array, real or complex
        
    Keyword Arguments:
          background : ndaarray
                       2D numpy array (real), backround hologram to be 
                       subtracted (default = None)
          normalise  : ndarray
                       2D numpy array (real), background hologram to be 
                       divided (default = None)
          window     : ndarray
                       2D numpy array (real), window to smooth edges. Will be  
                       resized to match size of img if necessary. 
                       (default = None)
          downsample : float
                       factor to downsample image by (default = 1)
    """    
    
    # Ensure the input hologram is a float or a complex float          
    if np.iscomplexobj(img):
        if precision == 'double': 
            imType = 'complex128'
        else:
            imType = 'complex64'
    else:
        if precision == 'double':
            imType = 'float64' 
        else:
            imType = 'float32'
    
    if img.dtype != imType:
        img = img.astype(imType)  
        
    if background is not None:
        if background.dtype != imType:
            background = background.astype(imType)  

    if normalise is not None:
        if normalise.dtype != imType:
            normalise = normalise.astype(imType)      
       
    # Background subtraction  
    if background is not None and np.shape(background) == np.shape(img):       
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
    if downsample != 1 and not np.iscomplexobj(img):                
        img = cv.resize(img, (int(np.shape(img)[1]/ downsample / 2) * 2, int(np.shape(img)[0] / downsample /2) *2 )   )
         
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

  


def fourier_plane_display(img):
    """ Returns a log-scale Fourier plane for display purposes.
    
    Returns a 2D numpy array, real.
    
    Parameters:
          img        :  ndarray
                        2D numpy array, real or complex.
    """
    
    cameraFFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    
    return cameraFFT    
