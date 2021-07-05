# -*- coding: utf-8 -*-
"""
PyHoloscope

Mike Hughes, Applied Optics Group, University of Kent
"""

import numpy as np
import cv2 as cv
import math
import scipy
from matplotlib import pyplot as plt

# Creates Fourier domain propagator for angular spectrum meethod. GridSize
# is size of image (in pixels) to refocus.
def propagator(gridSize, pixelSize, wavelength, depth):
    
    area = gridSize * pixelSize

    (xM, yM) = np.meshgrid(range(gridSize), range(gridSize))
    
    fac = wavelength/area;
    
    alpha = fac*(xM - gridSize/2 -1)
    beta = fac*(yM - gridSize/2 -1)
    
    prop = np.exp(-2*math.pi*1j*depth*np.sqrt(1 - alpha**2 - beta**2)/wavelength)
    
    prop[(alpha**2 + beta**2) > 1] = 0
    
    return prop

# Refocus using angular spectrum method. Takes a pre-computed propagator. Optionally specifiy 
# window - numpy array to multiple with prior to refocus, background - background image to subtract
# to make contrast hologram. Returns the complex image.
def refocus(img, propagator, **kwargs):
    background = kwargs.get('background', 0)
    window = kwargs.get('window', -1)
    
    if np.size(background) > 1:
        cHologram = img.astype('float32') - background.astype('float32')
    else:
        cHologram  = img.astype('float32')

    
    # If window was specified then multiple by window first
    if np.size(window) > 1:
        cHologram = cHologram * window
    
    return np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(cHologram)) * propagator))

    
    pass

# Produce a circular cosine window mask on grid of imgSize * imgSize. Mask
# is 0 for radius > circleSize and 1 for radius < (circleSize - skinThickness)
# The intermediate region is a smooth cosine function.
def circCosineWindow(imgSize, circleRadius, skinThickness):
    innerRad = circleRadius - skinThickness
    xM, yM = np.meshgrid(range(imgSize),range(imgSize))
    imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
    mask =  np.cos(math.pi / (2 * skinThickness) * (imgRad - innerRad))**2
    mask[imgRad < innerRad ] = 1
    mask[imgRad > innerRad + skinThickness] = 0
    return mask

# Returns score of how 'in focus' an image is based on selected method.
# Brenner, Sobel or Peak
def focusScore(img, method, **kwargs):
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
        scoreMap = cv.Sobel(img, ddepth = cv.CV_64F, dx=1, dy=1)
        focusScore = -np.mean(scoreMap)
    
    return focusScore

# Refocuses an image to specificed depth and returns focus score, Used by
# findFocus
def refocusAndScore(depth, img, pixelSize, wavelength, method):
    
    prop = propagator(np.shape(img)[0], pixelSize, wavelength, depth )
    refocImg = np.abs(refocus(img, prop))
    
    score = focusScore(refocImg, method)
    
    #plt.figure()
    #plt.imshow(refocImg, cmap='gray')
    #print("Wavelength:", wavelength, ", Pixel Size:", pixelSize)
    #print("Depth: ", depth, ", Score: ", score)
    
    return score


# Determine optimal depth to maximise sharpness in img
def findFocus(img, wavelength, pixelSize, depthRange, method, **kwargs):
        
    background = kwargs.get('background', 0)
    window = kwargs.get('window', -1)
    
    if np.size(background) > 1:
        cHologram = img.astype('float32') - background.astype('float32')
    else:
        cHologram  = img.astype('float32')

   
    if np.size(window) > 1:
        cHologram = cHologram * window
    #depth =  scipy.optimize.golden(refocusAndScore, brack = depthRange, args= (cHologram ,pixelSize, wavelength, method) )
    x0 = ((depthRange[0] + depthRange[1])/2,)
    bounds = ((depthRange[0],depthRange[1]),)
    depth =  scipy.optimize.minimize(refocusAndScore, x0 = x0, method = 'L-BFGS-B', bounds = bounds, args= (cHologram ,pixelSize, wavelength, method) )
    
    return depth.x[0]  
