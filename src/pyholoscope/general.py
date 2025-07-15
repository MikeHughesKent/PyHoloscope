# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

This file contains general functions, mostly for handling and displaying phase
maps.

"""

import warnings

import numpy as np
import cv2 as cv


def pre_process(img, background = None, normalise = None, window = None, downsample = 1., numba = False, precision = 'single'):
    """ Carries out steps required prior to refocusing.  
    This includes background correction, normalisation and windowing and 
    downsampling. It also coverts image to either type depending in specified
    precision (default is single precision).
    
    Parameters:
          img        : numpy.ndarray
                       raw hologram, 2D array, real or complex
        
    Keyword Arguments:
          background : numpy.ndaarray
                       backround hologram to be subtracted
                       2D real array (default = None)
          normalise  : numpy.ndarray
                       background hologram to be divided 
                       2D real array (default = None)
          window     : numpy.ndarray
                       window to smooth edges. 2D real array. Will be  
                       resized to match size of img if necessary. 
                       (default = None)
          downsample : float
                       factor to downsample image by (default = 1)
          numba      : bool
                       flag to use numba for speed up (default = False)
          precision  : str
                       'single' or 'double' (default = 'single')
    Returns:    
          numpy.ndarray  : pre-processed hologram, 2D array, real or complex

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
    """ Returns a real, log-scale Fourier plane for display purposes.
    
    Arguments:
          img             : numpy.ndarray
                            2D numpy array, real or complex.
    Returns:
           numpy.ndarray  : 2D real array, log scale of the Fourier plane.
           
    """
    
    fourierPlane = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    
    return fourierPlane   
