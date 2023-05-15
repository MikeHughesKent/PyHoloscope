# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

@author: Mike Hughes, Applied Optics Group, University of Kent

This file contains functions relaatd to numrical refocusing.

"""
import math
import numpy as np
try:
    import cupy as cp
    cudaAvailable = True
except:
    cudaAvailable = False
from PIL import Image
import cv2 as cv
import scipy
import time

from pyholoscope.focus_stack import FocusStack
from pyholoscope.focusing_numba import propagator_numba
from pyholoscope.roi import Roi
import pyholoscope.general 
from pyholoscope.utils import dimensions


def propagator(gridSize, wavelength, pixelSize, depth, geometry = 'plane', precision = 'single'):
    """ Creates Fourier domain propagator for angular spectrum method. 
    Returns the propagator as a complex 2D numpy array.
    
    Speed up by only calculating top left quadrant and then duplicating 
    (with flips) to create the other quadrants. 
    
    Arguments:
        gridSize   : float, size of square image (in pixels) to refocus, or
                     tuple of (float, float), size of rectangular image.
        pixelSize  : float, physical size of pixels
        wavelength : float, in same units as pixelSize
        depth      : float, refocus depth in same units as pixelSize
    
    Optional Keyword Arguments:
        geometry   : str, 'plane' (default) or 'point'
        precision  : str, numerical precision of ouptut, 'single' (defualt) 
                     or 'double'

    """
    #assert gridSize % 2 == 0, "Grid size must be even"
    
    if precision == 'double':
        dataType = 'complex128'
    else:
        dataType = 'complex64'
    
    
    gridWidth, gridHeight = dimensions(gridSize)
    
    width = gridWidth * pixelSize
    height = gridHeight * pixelSize

    centreX = int(gridWidth//2)
    centreY = int(gridHeight//2)

    # Points to generate one quadrant of propagator on
    (xM, yM) = np.meshgrid(range(centreX + 1 ), range(centreY + 1) )
    
    
    delta0x = 1/width 
    delta0y = 1/height   
 
    # Generate one quadrant of the propagator
    if geometry == 'point':
        u = delta0x*xM
        v = delta0y*yM
        propCorner = np.exp(1j*math.pi*wavelength*depth*(u**2 + v**2))
    elif geometry == 'plane':        
        alpha = wavelength * xM/width
        beta = wavelength * yM/height
        propCorner = np.exp( (1j * 2 * math.pi * depth * np.sqrt(1 - alpha**2 - beta**2) /wavelength ))
        propCorner[alpha**2 + beta**2 > 1] = 0  
    else:
        raise Exception("Invalid geometry.")               
    
    # For full propagator
    prop = np.zeros((gridHeight, gridWidth), dtype = dataType)
        
    # Duplicate the top left quadrant into the other three quadrants as
    # this is quicker then explicitly calculating the values
    prop[:centreY + 1, :centreX + 1] = propCorner                      # top left
    prop[:centreY + 1, centreX:] = (np.flip(propCorner[:, 1:],1) )     # top right
    prop[centreY:, :] = (np.flip(prop[1:centreY + 1, :],0))            # bottom left

    return prop


def refocus(img, propagator, **kwargs):    
    """ 
    Refocus a hologram using the angular spectrum method. 
    
    Arguments:
        img           : 2D numpy array, raw hologram.  
        propagator    : 2D numpy array, as returned from propagator().
    
    Optional Keyword Arguments:
        FourierDomain : boolean, if True then img is assumed to be already the
                        FFT of the hologram, useful for speed when performing
                        multiple refocusing of the same hologram. (default =
                                                                   False)
        cuda          : boolean, if True GPU will be used if available.
        others        : pass any keyword arguments from pre_process() to 
                        apply this pre-processing prior to refocusing
    """
       
    imgIsFourier = kwargs.pop('FourierDomain', False)
    cuda = kwargs.pop('cuda', True)
    
    assert np.shape(img) == np.shape(propagator), "Propagator is the wrong size."
    
    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if imgIsFourier:  
        if cuda is True and cudaAvailable is True:
            if type(img) is np.ndarray:
                img = cp.array(img)
            if type(propagator) is np.ndarray:
                propagator = cp.array(propagator)
            return cp.asnumpy(cp.fft.ifft2(img * propagator))
        else:
            return scipy.fft.ifft2(img * propagator)
     
    else:   # If we are sent the spatial domain image
        cHologram = pyholoscope.pre_process(img, **kwargs)
        if cuda is True and cudaAvailable is True:
                if type(cHologram) is np.ndarray:
                    cHologram = cp.array(cHologram)
                if type(propagator) is np.ndarray:
                    propagator = cp.array(propagator)
                return cp.asnumpy(cp.fft.ifft2(cp.fft.fft2(cHologram) * propagator))
        else:
                return scipy.fft.ifft2(scipy.fft.fft2(cHologram) * propagator)
           
   
def focus_score(img, method):
    """ Returns score of how 'in focus' an amplitude image is.
    
    Score is returned as a float, the lower the better the focus.
    
    Arguments:
        img     : 2D numpy array, real, image
        method  : str, scoring method, options are: Brenner, Sobel, 
                         SobelVariance, Var, DarkFcous or Peak
                      
    """

    focusScore = 0
    
    if method == 'Brenner':        
        (h,w) = np.shape(img)
        BrennX = np.zeros((h, w))
        BrennY = np.zeros((h, w))
        BrennX[0:-2,:] = img[2:,:]-img[0:-2,] 
        BrennY[:,0:-2] = img[:,2:]-img[:,0:-2] 
        scoreMap = np.maximum(BrennY**2, BrennX**2)    
        focusScore = -np.mean(scoreMap)
           
    elif method == 'Peak':
        focusScore = -np.max(img)    
    
    elif method == 'Sobel':
        filtX = np.array( [ [ 1, 0, -1] , [ 2, 0, -2], [ 1,  0, -1]] )
        filtY = np.array( [ [ 1, 2,  1] , [ 0, 0,  0], [-1, -2, -1]] )
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focusScore = -np.mean(sobel)
        
    elif method == 'SobelVariance':
        filtX = np.array( [ [ 1, 0, -1] , [ 2, 0, -2], [ 1,  0, -1]] )
        filtY = np.array( [ [ 1, 2,  1] , [ 0, 0,  0], [-1, -2, -1]] )
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focusScore = -(np.std(sobel)**2)
    
    elif method == 'Var':
        focusScore = -np.std(img)
                
    # https://doi.org/10.1016/j.optlaseng.2020.106195
    elif method == 'DarkFocus':
        kernelX = np.array([[-1,0,1]])
        kernelY = kernelX.transpose()
        gradX = cv.filter2D(img, -1, kernelX)
        gradY = cv.filter2D(img, -1, kernelY)
        mean, stDev = cv.meanStdDev(gradX**2 + gradY**2)
        focusScore = -(stDev[0,0]**2)
    
    else:
        raise Exception("Invalid scoring method.")
        
    return focusScore


def refocus_and_score(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT, useNumba, useCuda, precision = 'single'):
    """ Refocuses an image to specificed depth and returns focus score, used by
    findFocus.
    """
    
    # Whether we are using a look up table of propagators or calculating it each time  
    if propLUT is None:
        if useNumba:
            prop = propagator_numba(np.shape(imgFFT)[0], wavelength, pixelSize, depth, precision = precision)
        else:
            prop = propagator(np.shape(imgFFT)[0], wavelength, pixelSize, depth, precision = precision)

    else:
        prop = propLUT.propagator(depth)        
    
    # We are working with the FFT of the hologram for speed 
    refocImg = np.abs(refocus(imgFFT, prop, FourierDomain = True, cuda = useCuda))

    # If we are running focus metric only on a ROI, extract the ROI
    if scoreROI is not None:  
        refocImg = scoreROI.crop(refocImg)
    
    score = focus_score(refocImg, method)

    return score


def find_focus(img, wavelength, pixelSize, depthRange, method, **kwargs):
    """ Determine the refocus depth which maximises the focus metric on image
    """   
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    scoreROI = kwargs.get('roi', None)
    margin = kwargs.get('margin', None)  
    propLUT = kwargs.get('propagatorLUT', None)
    coarseSearchInterval = kwargs.get('coarseSearchInterval', None)
    useNumba = kwargs.get('numba', False)
    useCuda = kwargs.get('cuda', False)
    
    cHologram = pyholoscope.pre_process(img, background=background, window = window)
    
    # If a margin is specified, this means we only refocus the ROI plus a 
    # margin around it for speed. Define the ROI here.
    if margin is not None and scoreROI is not None:
        refocusROI = Roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])
        scoreROI = Roi(margin, margin, scoreROI.width, scoreROI.height)
    else:
        refocusROI = None
   
    # If we are only refocusing around a ROI, crop to the ROI + margn    
    if refocusROI is not None:
        cropImg = refocusROI.crop(cHologram)
        scoreROI.constrain(0,0,np.shape(cropImg)[0], np.shape(cropImg)[1])        
    else:
        cropImg = cHologram   # otherwise use whole image
        
    # Pre-compute the FFT of the hologram since we need this for every trial depth    
    imgFFT = scipy.fft.fft2(cropImg)
    
    if coarseSearchInterval is not None:
        startDepth = coarse_focus_search(imgFFT, depthRange, coarseSearchInterval, pixelSize, wavelength, method, scoreROI, propLUT)
        intervalSize = (depthRange[1] - depthRange[0]) / coarseSearchInterval
        minBound = max(depthRange[0], startDepth - intervalSize)
        maxBound = min(depthRange[1], startDepth + intervalSize)
        depthRange = [minBound, maxBound]
    else:
        startDepth = (max(depthRange) - min(depthRange))/2

    # Find the depth using optimiser
    depth = scipy.optimize.minimize_scalar(refocus_and_score, method = 'Bounded', bounds = depthRange, bracket = depthRange, options = {'xtol': 0.00001, 'maxiter': 20}, args= (imgFFT ,pixelSize, wavelength, method, scoreROI, propLUT, useNumba, useCuda) )

    return depth.x 


def coarse_focus_search(imgFFT, depthRange, nIntervals, pixelSize, wavelength, method, scoreROI, propLUT):
    """ An initial check for approximate location of good focus depths prior to a finer search. Called
     by findFocus    
    """
    searchDepths = np.linspace(depthRange[0], depthRange[1], nIntervals)
    focusScore = np.zeros_like(searchDepths)
    for idx, depth in enumerate(searchDepths):
        focusScore[idx] = refocus_and_score(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT)
    
    bestInterval = np.argmin(focusScore)
    bestDepth = searchDepths[bestInterval]
   
    return bestDepth
    
    
def focus_score_curve(img, wavelength, pixelSize, depthRange, nPoints, method, **kwargs):
    """ Produce a plot of focus score against depth, mainly useful for debugging
    erroneous focusing
    """     
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    scoreROI = kwargs.get('roi', None)
    margin = kwargs.get('margin', None)
    precision = kwargs.get('precision', None)

    
    
    if background is not None:
        cHologram = img.astype('float32') - background.astype('float32')
    else:
        cHologram  = img.astype('float32')     
        
    if margin is not None and scoreROI is not None:
        refocusROI = Roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])                
    else:
        refocusROI = None
   
    if window is not None:
        cHologram = cHologram * window
                
    if margin is not None and scoreROI is not None:
        refocusROI = Roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])                
        cropImg = refocusROI.crop(cHologram)
    else:
        cropImg = cHologram
    
    # Do the forwards FFT once for speed
    cHologramFFT = scipy.fft.fft2(cropImg)
    
    score = list()
    depths = np.linspace(depthRange[0], depthRange[1], nPoints)
    for idx, depth in enumerate(depths):
        score.append(refocus_and_score(depth, cHologramFFT, pixelSize, wavelength, method, scoreROI,  None, precision = precision))
        
    return score, depths


def refocus_stack(img, wavelength, pixelSize, depthRange, nDepths, **kwargs):
    """ Numerical refocusing of a hologram to produce a depth stack. 'depthRange' is a tuple
    defining the min and max depths, the resulting stack will have 'nDepths' images
    equally spaced between these limits. 
    """
    window = kwargs.get('window', None)
    useNumba = kwargs.get('numba', False)
    background = kwargs.get('preBackground', None)
    precision = kwargs.get('precision', 'single')
    
    depths = np.linspace(depthRange[0], depthRange[1], nDepths)    
    
    # Apply pre-processing and then take 2D FFT
    cHologram = pyholoscope.pre_process(img, **kwargs)
    cHologramFFT = (scipy.fft.fft2(cHologram))
    
    # Tell refocus that we are providing the FFT
    kwargs['FourierDomain'] = True
    
    imgStack = FocusStack(cHologramFFT, depthRange, nDepths)

    for idx, depth in enumerate(depths):
        if useNumba:
            prop = propagator_numba(np.shape(img)[1::-1], wavelength, pixelSize, depth, precision = precision)
        else:
            prop = propagator(np.shape(img)[1::-1], wavelength, pixelSize, depth, precision = precision)
      
        imgStack.add_idx( pyholoscope.pre_process( refocus(cHologramFFT, prop, **kwargs), window=window), idx)

    return imgStack


