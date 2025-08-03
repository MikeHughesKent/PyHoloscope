# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

This file contains general functions.

"""

import warnings

import numpy as np
import cv2 as cv
import scipy as sp

from numba import jit, njit, prange



def pre_process(
    img,
    background=None,
    normalise=None,
    window=None,
    downsample=1.0,
    numba=False,
    precision="single",
):
    """Carries out steps required prior to refocusing.
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
        if precision == "double":
            imType = "complex128"
        else:
            imType = "complex64"
    else:
        if precision == "double":
            imType = "float64"
        else:
            imType = "float32"

    if img.dtype != imType:
        img = img.astype(imType)

    if background is not None and not np.iscomplexobj(img):
        if background.dtype != imType:
            background = background.astype(imType)

    if normalise is not None and not np.iscomplexobj(img):
        if normalise.dtype != imType:
            normalise = normalise.astype(imType)

    remove_background(img, background)
    normalise_hologram(img, normalise)
    apply_window(img, window)
    downsample_hologram(img, downsample)
    
    return img


def remove_background(img, background):
    """Removes the background from a hologram.

    Arguments:  
            img        : numpy.ndarray
                            raw hologram, 2D array, real or complex
            background : numpy.ndarray
                            background hologram to be subtracted, same shape as img
    Returns:
            numpy.ndarray  : hologram with background removed, 2D array, real or complex
    """

    if background is not None and np.shape(background) == np.shape(img):
        if np.iscomplexobj(img):
            imgAmp = np.abs(img)
            imgPhase = np.angle(img)
            img = (imgAmp - np.sqrt(background)) * np.exp(1j * imgPhase)           
        else:
            img = img - background

    return img

def normalise_hologram(img, ref_img):
    """Normalises a hologram by dividing it by a reference image.

    Arguments:
            img        : numpy.ndarray
                            raw hologram, 2D array, real or complex
            ref_img    : numpy.ndarray
                            reference image to be divided by, same shape as img
    Returns:
            numpy.ndarray  : normalised hologram, 2D array, real or complex
    """

    if ref_img is not None and np.shape(ref_img) == np.shape(img):
        if np.iscomplexobj(img):
            imgAmp = np.abs(img)
            imgPhase = np.angle(img)
            img = (imgAmp / np.sqrt(ref_img)) * np.exp(1j * imgPhase)          
        else:
            img = img / ref_img

    return img

def apply_window(img, window, allow_resize=True):
    """Applies a window to an image. If the window is not the same size as the image, it will be resized to match.
    
    Arguments:
            img        : numpy.ndarray
                         image to apply window to, 2D array, real or complex
            window     : numpy.ndarray
                         window to apply, 2D real array
    Keyword Arguments:
            allow_resize : bool
                           if True, allows the window to be resized to match the image size (default = True). Otherwise,
                           if the window is not the same size as the image, the function will return the image unchanged.    
    Returns:
            None
    """
        
    if window is not None:

        # If the window is the wrong size, reshape it to match hologram
        if np.shape(window) != np.shape(img):
            if allow_resize:
                warnings.warn("Window needed resizing, may effect processing speed.")
                window = cv.resize(window, (np.shape(img)[1], np.shape(img)[0]))
            else:
                return img

        if np.iscomplexobj(img):
            img.imag = img.imag * window
            img.real = img.real * window
            img[img == -0 + 0j] = 0j  # Otherwise phase angle looks weird when plotted
        else:
            img = img * window

def downsample_hologram(img, factor):
    """Downsamples a hologram by a factor of 'factor'. 
    Arguments:
            img        : numpy.ndarray
                         raw hologram, 2D array, real or complex
            factor     : int
                         factor to downsample by, should be an integer (default = 1)
    Returns:
            numpy.ndarray  : downsampled hologram, 2D array, real or complex
    """
    
    if factor != 1 and not np.iscomplexobj(img):
        img = cv.resize(
                img,
                (
                    int(np.shape(img)[1] / factor / 2) * 2,
                    int(np.shape(img)[0] / factor / 2) * 2,
                ),
            )
    return img

def fourier_plane_display(img):
    """Returns a real, log-scale Fourier plane for display purposes.

    Arguments:
          img             : numpy.ndarray
                            2D numpy array, real or complex.
    Returns:
           numpy.ndarray  : 2D real array, log scale of the Fourier plane.

    """

    fourierPlane = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))

    return fourierPlane

def fast_full_2D_fft(input, use_numba=True):
    """"
    Computes the full 2D FFT of a real input array using the rfft2 function and reconstructs the full FFT 
    from the real FFT output by exploiting the symmetry. May be faster than np/sp fft2 for large arrays.

    Arguments:
          input : numpy.ndarray 
                  2D numpy array, real
    Keyword Arguments:
          use_numba : bool
                      If True, uses numba for speed up (default = True). If this is set to False, this function
                      is likely to be slower than np.fft.fft2 or sp.fft.fft2.
    Returns:    
            numpy.ndarray : 2D complex numpy array, full FFT of the input.
            
    """
    
    M, N = input.shape

    rfft = sp.fft.rfft2(input)
    

    if use_numba:
        #pass
        out = reconstruct_full_fft2(rfft, M, N)
    else:
        # Copy the rfft2 output into the left half
        mid = N//2
        out = np.zeros((M, N), dtype='complex128')
        out[:, :mid+1] = rfft
        
        if N % 2 == 0:
            reflected = np.conj(np.flipud(np.fliplr(rfft[1:, 1:])))
            out[1:, mid:] = reflected
            out[0, mid+1:] = np.conj(rfft[0,mid-1:0:-1])
        else:
            reflected = np.conj(np.flipud(np.fliplr(rfft[1:, 1:])))
            out[1:, mid+1:] = reflected
            out[0, mid+1:] = np.conj(rfft[0,mid:0:-1])

    return out

