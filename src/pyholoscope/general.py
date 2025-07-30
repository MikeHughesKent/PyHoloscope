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

    # Background subtraction
    if background is not None and np.shape(background) == np.shape(img):
        if np.iscomplexobj(img):
            imgAmp = np.abs(img)
            imgPhase = np.angle(img)
            img = (imgAmp - np.sqrt(background)) * np.exp(1j * imgPhase)           
        else:
            img = img - background

    # Background normalisation
    if normalise is not None:
        if np.iscomplexobj(img):
            imgAmp = np.abs(img)
            imgPhase = np.angle(img)
            img = (imgAmp / np.sqrt(normalise)) * np.exp(1j * imgPhase)          
        else:
            img = img / normalise

    # Apply window
    if window is not None:
        # If the window is the wrong size, reshape it to match hologram
        if np.shape(window) != np.shape(img):
            warnings.warn("Window needed resizing, may effect processing speed.")
            window = cv.resize(window, (np.shape(img)[1], np.shape(img)[0]))

        if np.iscomplexobj(img):
            img.imag = img.imag * window
            img.real = img.real * window
            img[img == -0 + 0j] = 0j  # Otherwise phase angle looks weird when plotted
        else:
            img = img * window

    # Apply downsampling
    if downsample != 1 and not np.iscomplexobj(img):
        img = cv.resize(
            img,
            (
                int(np.shape(img)[1] / downsample / 2) * 2,
                int(np.shape(img)[0] / downsample / 2) * 2,
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



@jit(nopython=True, parallel=True)
def reconstruct_full_fft2(rfft, M, N):
    """ Used by fast_full_2D_fft() to reconstruct the full 2D FFT from the real FFT output.
    """
    out = np.zeros((M, N), dtype=np.complex128)
    mid = N // 2

    # Copy the rfft2 output into the left half
    for i in prange(M):
        for j in prange(mid + 1):
            out[i, j] = rfft[i, j]

    # Fill the symmetric (redundant) part manually
    for i in prange(M):
        for j in prange(mid + 1, N):
            ii = (-i) % M
            jj = (-j) % N
            z = out[ii, jj]
            out[i, j] = complex(z.real, -z.imag)

    return out