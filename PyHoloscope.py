# -*- coding: utf-8 -*-
"""
PyHoloscope

Mike Hughes, Applied Optics Group, University of Kent
"""

import numpy as np
import cv2 as cv
import math
import scipy
import time
from matplotlib import pyplot as plt

# Creates Fourier domain propagator for angular spectrum meethod. GridSize
# is size of image (in pixels) to refocus.
def propagator(gridSize, wavelength, pixelSize, depth):
    
    
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
    imgIsFourier = kwargs.get('FourierDomain', False)
    
    
    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if imgIsFourier:  
        
       return np.fft.ifft2(np.fft.fftshift(img * propagator))

    else:   # If we are sent the spatial domain image
        
        if np.size(background) > 1:
            cHologram = img.astype('float32') - background.astype('float32')
        else:
            cHologram  = img.astype('float32')
    
        # If window was specified then multiply by window first
        if np.size(window) > 1:
            cHologram = cHologram * window
    
        return np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(cHologram)) * propagator))

    
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
        #plt.figure()
        #plt.imshow(img, cmap='gray')
        #plt.figure()
        #plt.imshow(scoreMap, cmap='gray')
    
    if method == 'Peak':
        focusScore = -np.max(img)
    
    if method == 'Sobel':
        filtX = np.array( [ [ 1, 0, -1] , [ 2, 0, -2], [ 1,  0, -1]] )
        filtY = np.array( [ [ 1, 2,  1] , [ 0, 0,  0], [-1, -2, -1]] )
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focusScore = -np.mean(sobel)
    
    if method == 'Var':
        focusScore = np.std(img)
    
    return focusScore

# Refocuses an image to specificed depth and returns focus score, used by
# findFocus
def refocusAndScore(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT):
    
    # Whether we are using a look up table of propagators or calclating it each time  
    if propLUT is None:
        prop = propagator(np.shape(imgFFT)[0], wavelength, pixelSize, depth)
    else:
        prop = propLUT.propagator(depth)        
    
    # We are working with the FFT of the hologram    
    #t0 = time.time()
    refocImg = np.abs(refocus(imgFFT, prop, FourierDomain = True))
    #print("FFT Time: ", time.time() - t0)
    if scoreROI is not None:  
        refocImg = scoreROI.crop(refocImg)
    
    #t0 = time.time()
    score = focusScore(refocImg, method)
    #print("Score Time: ", time.time() - t0)

    #plt.figure()
    #plt.imshow(refocImg, cmap='gray')
    #print("Wavelength:", wavelength, ", Pixel Size:", pixelSize)
    #print("Depth: ", depth, ", Score: ", score)
    
    return score




# Determine optimal depth to maximise sharpness in img
def findFocus(img, wavelength, pixelSize, depthRange, method, **kwargs):
        
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    scoreROI = kwargs.get('roi', None)
    margin = kwargs.get('margin', None)  
    propLUT = kwargs.get('propagatorLUT', None)
    
    if background is not None:
        cHologram = img.astype('float32') - background.astype('float32')
    else:
        cHologram  = img.astype('float32')

    if margin is not None and scoreROI is not None:
        refocusROI = roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])
        scoreROI = roi(margin, margin, scoreROI.width, scoreROI.height)
    else:
        refocusROI = None
   
    if window is not None:
        cHologram = cHologram * window
        
    if refocusROI != None:
        cropImg = refocusROI.crop(cHologram)
        scoreROI.constrain(0,0,np.shape(cropImg)[0], np.shape(cropImg)[1])
        
        print(scoreROI)
    else:
        cropImg = cHologram
        
    # Pre-compute the FFT of the hologram since we need this for every trial depth    
    imgFFT = np.fft.fftshift(np.fft.fft2(cropImg))
    depth =  scipy.optimize.minimize_scalar(refocusAndScore, method = 'bounded', bounds = depthRange, args= (imgFFT ,pixelSize, wavelength, method, scoreROI, propLUT) )

    return depth.x 





# Produce a plot of focus score against depth
def focusScoreCurve(img, wavelength, pixelSize, depthRange, nPoints, method, **kwargs):
        
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    scoreROI = kwargs.get('roi', None)
    margin = kwargs.get('margin', None)
    
    if background is not None:
        cHologram = img.astype('float32') - background.astype('float32')
    else:
        cHologram  = img.astype('float32')
    #plt.figure()
    #plt.imshow(cHologram, cmap='gray')     
        
    if margin is not None and scoreROI is not None:
        refocusROI = roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])                
    else:
        refocusROI = None
   
    if window is not None:
        cHologram = cHologram * window
                
    if margin is not None and scoreROI is not None:
        refocusROI = roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])                
        cropImg = refocusROI.crop(cHologram)
    else:
        cropImg = cHologram
    
    # Do the forwards FFT once for speed
    cHologramFFT = np.fft.fftshift(np.fft.fft2(cropImg))
    
    score = list()
    depths = np.linspace(depthRange[0], depthRange[1], nPoints)
    for idx, depth in enumerate(depths):
        score.append(refocusAndScore(depth, cHologramFFT, pixelSize, wavelength, method, scoreROI,  None))
        
    return score, depths



# Utility class for region of interest
class roi:
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.width) + ',' + str(self.height)
    
    # Stops ROI exceeding images size
    def constrain(self, minX, minY, maxX, maxY):
        self.x = max(self.x, minX)
        self.y = max(self.y, minY)

        self.width = min(self.width, maxX - minX + 1) 
        self.height = min(self.width, maxY - minY + 1)
        
    # img is cropped to ROI    
    def crop (self, img):
        return img[self.x: self.x + self.width - 1, self.y:self.y + self.height - 1]
    

# LUT of propagators
class PropLUT:
    def __init__(self, imgSize, wavelength, pixelSize, depthRange, nDepths):
        self.depths = np.linspace(depthRange[0], depthRange[1], nDepths)
        self.nDepths = nDepths
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        self.propTable = np.zeros((nDepths, imgSize, imgSize), dtype = 'complex128')
        for idx, depth in enumerate(self.depths):
            self.propTable[idx,:,:] = propagator(imgSize, wavelength, pixelSize, depth)
            
    def __str__(self):
        return "LUT of " + str(self.nDepths) + " propagators from depth of " + str(self.depths[0]) + " to " + str(self.depths[-1]) + ". Wavelength: " + str(self.wavelength) + ", Pixel Size: " + str(self.pixelSize)
            
    def propagator(self, depth): 
        
        # Find nearest propagator
        if depth < self.depths[0] or depth > self.depths[-1]:
            return - 1
        idx = round((depth - self.depths[0]) / (self.depths[-1] - self.depths[0]) * self.nDepths)
        #print("Desired Depth: ", depth, "Used Depth:", self.depths[idx])
        return self.propTable[idx, :,:]
        
        
        
        
        