# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Mike Hughes, Applied Optics Group, University of Kent

PyHoloscope is a python library for holographic microscopy.

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

from pyholoscope.focus_stack import FocusStack
from pyholoscope.focusing_numba import propagator_numba
from pyholoscope.roi import Roi
import pyholoscope.general 


def propagator_slow(gridSize, wavelength, pixelSize, depth):
    """ 
    Slow version of propagator. Creates Fourier domain propagator for angular 
    spectrum meethod. GridSize is size of image (in pixels) to refocus. Retained only
    for testing purposes, use propagator instead.
    """
    
    area = gridSize * pixelSize

    (xM, yM) = np.meshgrid(range(gridSize), range(gridSize))
    
    delta0 = 1/area
    u = delta0*(xM - gridSize/2 +.5)
    v = delta0*(yM - gridSize/2 +.5)
    prop= np.exp(1j*math.pi*wavelength*depth*(u**2 + v**2))

    
    return prop


def propagator(gridSize, wavelength, pixelSize, depth):
    """ Creates Fourier domain propagator for angular spectrum meethod. Speeds
    up process by only calculating top left quadrant and then duplicating (with flips)
    to create the other three quadrants.
    """
    assert gridSize % 2 == 0, "Grid size must be even"
    
    area = gridSize * pixelSize
    midPoint = int(gridSize/2)

    (xM, yM) = np.meshgrid(range(int(gridSize/2)), range(int(gridSize/2)))
    
    delta0 = 1/area
    u = delta0*(xM - gridSize/2 +.5)
    v = delta0*(yM - gridSize/2 +.5)
    propCorner = np.exp(1j*math.pi*wavelength*depth*(u**2 + v**2))
    
    prop = np.zeros((gridSize, gridSize), dtype ='complex64')
    prop[:midPoint, :midPoint] = propCorner
    prop[midPoint:, :midPoint] = np.flip(propCorner,0)
    prop[:, midPoint:] = np.flip(prop[:, :midPoint],1)

    return prop


def refocus(img, propagator, **kwargs):    
    """ Refocus using angular spectrum method. Takes a hologram (with any pre-processing
    such as background removal already performed) and a pre-computed propagator. 
    """
    
    imgIsFourier = kwargs.get('FourierDomain', False)
    cuda = kwargs.get('cuda', True)
    if np.shape(img) != np.shape(propagator):
        return None
    
    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if imgIsFourier:  
        if cuda is True and cudaAvailable is True:
            img2 = cp.array(img)
            propagator2 = cp.array(propagator)
           
            return cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(img2 * propagator2)))
        else:
            return np.fft.ifft2(np.fft.fftshift(img * propagator))

    else:   # If we are sent the spatial domain image
        cHologram = pyholoscope.pre_process(img, **kwargs)
        if cuda is True and cudaAvailable is True:
            cHologram2 = cp.array(cHologram)
            propagator2 = cp.array(propagator)
            return cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(cp.fft.fftshift(cp.fft.fft2((cHologram2))) * propagator2)))
        else:
            return np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2((cHologram))) * propagator))

   
def focus_score(img, method):
    """ Returns score of how 'in focus' an image is based on selected method.
    Method options are: Brenner, Sobel, SobelVariance, Var, DarkFcous or Peak
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
       
    
    if method == 'Peak':
        focusScore = -np.max(img)
    
    
    if method == 'Sobel':
        filtX = np.array( [ [ 1, 0, -1] , [ 2, 0, -2], [ 1,  0, -1]] )
        filtY = np.array( [ [ 1, 2,  1] , [ 0, 0,  0], [-1, -2, -1]] )
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focusScore = -np.mean(sobel)
        
    if method == 'SobelVariance':
        filtX = np.array( [ [ 1, 0, -1] , [ 2, 0, -2], [ 1,  0, -1]] )
        filtY = np.array( [ [ 1, 2,  1] , [ 0, 0,  0], [-1, -2, -1]] )
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focusScore = -(np.std(sobel)**2)
    
    if method == 'Var':
        focusScore = np.std(img)
                
    # https://doi.org/10.1016/j.optlaseng.2020.106195
    if method == 'DarkFocus':
        kernelX = np.array([[-1,0,1]])
        kernelY = kernelX.transpose()
        gradX = cv.filter2D(img, -1, kernelX)
        gradY = cv.filter2D(img, -1, kernelY)
        mean, stDev = cv.meanStdDev(gradX**2 + gradY**2)
        
        focusScore = -(stDev[0,0]**2)
        
    return focusScore


def refocus_and_score(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT):
    """ Refocuses an image to specificed depth and returns focus score, used by
    findFocus
    """
    
    # Whether we are using a look up table of propagators or calclating it each time  
    if propLUT is None:
        prop = propagator(np.shape(imgFFT)[0], wavelength, pixelSize, depth)
    else:
        prop = propLUT.propagator(depth)        
    
    # We are working with the FFT of the hologram for speed 
    refocImg = np.abs(refocus(imgFFT, prop, FourierDomain = True))

    # If we are running focus metric only on a ROI, extract the ROI
    if scoreROI is not None:  
        refocImg = scoreROI.crop(refocImg)
    
    score = focus_score(refocImg, method)
    #print(depth, score)
    #print(time.perf_counter() - t1)
    return score


def find_focus(img, wavelength, pixelSize, depthRange, method, **kwargs):
    """ Determine the refocus depth which maximise the focus metric on image
    """   
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    scoreROI = kwargs.get('roi', None)
    margin = kwargs.get('margin', None)  
    propLUT = kwargs.get('propagatorLUT', None)
    coarseSearchInterval = kwargs.get('coarseSearchInterval', None)
    
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
    imgFFT = np.fft.fftshift(np.fft.fft2(cropImg))
    
    if coarseSearchInterval is not None:
        startDepth = coarse_focus_search(imgFFT, depthRange, coarseSearchInterval, pixelSize, wavelength, method, scoreROI, propLUT)
        intervalSize = (depthRange[1] - depthRange[0]) / coarseSearchInterval
        minBound = max(depthRange[0], startDepth - intervalSize)
        maxBound = min(depthRange[1], startDepth + intervalSize)
        depthRange = [minBound, maxBound]
    else:
        startDepth = (max(depthRange) - min(depthRange))/2

    # Find the depth using optimiser
    depth = scipy.optimize.minimize_scalar(refocus_and_score, method = 'bounded', bounds = depthRange, args= (imgFFT ,pixelSize, wavelength, method, scoreROI, propLUT) )

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
    print("best interval", bestInterval)
    print("best depth", bestDepth)
    
    return bestDepth
    
    
def focus_score_curve(img, wavelength, pixelSize, depthRange, nPoints, method, **kwargs):
    """ Produce a plot of focus score against depth, mainly useful for debugging
    erroneous focusing
    """     
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    scoreROI = kwargs.get('roi', None)
    margin = kwargs.get('margin', None)
    
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
    cHologramFFT = np.fft.fftshift(np.fft.fft2(cropImg))
    
    score = list()
    depths = np.linspace(depthRange[0], depthRange[1], nPoints)
    for idx, depth in enumerate(depths):
        score.append(refocus_and_score(depth, cHologramFFT, pixelSize, wavelength, method, scoreROI,  None))
        
    return score, depths


def refocus_stack(img, wavelength, pixelSize, depthRange, nDepths, **kwargs):

    """ Numerical refocusing of a hologram to produce a depth stack. 'depthRange' is a tuple
    defining the min and max depths, the resulting stack will have 'nDepths' images
    equally spaced between these limits. Specify 'imgisFFT' = true if the provided
    'img' is aready in Fourier domain.   
    """
    window = kwargs.get('window', None)
    useNumba = kwargs.get('numba', False)
    #cHologram = pyholoscope.pre_process(img, **kwargs)
    cHologram = img
    cHologramFFT = np.fft.fftshift(np.fft.fft2(cHologram))
    depths = np.linspace(depthRange[0], depthRange[1], nDepths)
    kwargs['FourierDomain'] = True
    imgStack = FocusStack(cHologramFFT, depthRange, nDepths)

    for idx, depth in enumerate(depths):
        if useNumba:
            prop = propagator_numba(np.shape(img)[0], wavelength, pixelSize, depth)
        else:
            prop = propagator(np.shape(img)[0], wavelength, pixelSize, depth)

        imgStack.add_idx(pyholoscope.post_process(refocus(cHologramFFT, prop, **kwargs), window=window), idx)

    return imgStack