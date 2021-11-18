# -*- coding: utf-8 -*-
"""
PyHoloscope

Mike Hughes, Applied Optics Group, University of Kent

PyHoloscope is a python library to assist with processing of holographic 
microscopy imaging. It is currently under development.

Classes:
    Holo - Reconstruction and Refocusing
    PropLUT - Propagator Look Up Table
    Roi - Region of interest
    FocusStack - Stack of images refocused to different depth
    

"""

import numpy as np
import cv2 as cv
import math
import scipy
import scipy.optimize

import time
from matplotlib import pyplot as plt
from PIL import Image
import time

def __init__():
    pass


INLINE_MODE = 1
OFFAXIS_MODE = 2

########## Holo Class #############

class Holo:
    

    
    
    def __init__(self, mode, wavelength, pixelSize, **kwargs):
        
        self.mode = mode
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        
        self.depth = kwargs.get('depth', 0)
        self.background = kwargs.get('background',None)
        self.window = kwargs.get('window', None)
        self.findFocusMethod = kwargs.get('findFocusMethod', 'Sobel')
        self.findFocusRoi = kwargs.get('findFocusRoi', None)
        self.findFocusMargin = kwargs.get('findFocusMargin', None)
       
        self.backgroundField = None
        self.propagatorDepth = 0
        self.propagatorWavelength = 0
        self.propagatorPixelSize = 0
        self.propagatorSize = 0
        self.propagator = None
        self.propagatorLUT = None
        
        self.relativePhase = False
        
        # Off-axis
        self.cropCentre = (0,0)
        self.cropRadius = 0
        
    def __str__(self):
        return "PyHoloscope Holo Class. Wavelength: " + str(self.wavelength) + ", Pixel Size: " + str(self.pixelSize)

    def setDepth(self, depth):
        self.depth = depth
        
    def setBackground(self, background):
        self.background  = background
      
        
    def clearBackground(self):
        self.background = None        
    
    def setWindow(self, img, circleRadius, skinThickness):
        self.window = circCosineWindow(np.shape(img)[0], circleRadius, skinThickness)
        
    def setOffAxisMod(self, cropCentre, cropRadius):
        self.cropCentre = cropCentre
        self.cropRadius = cropRadius
        
    def autoFindOffAxisMod(self):
        if self.background is not None:
            self.cropCentre = offAxisFindMod(self.background)
            self.cropRadius = offAxisFindCropRadius(self.background)
         
    
    def offAxisBackgroundField(self):
        self.backgroundField = offAxisDemod(self.background, self.cropCentre, self.cropRadius)
        self.relativePhase = True
                    
    def updatePropagator(self, img):
        self.propagator = propagator(np.shape(img)[0], self.wavelength, self.pixelSize, self.depth)
        self.propagatorWavelength = self.wavelength
        self.propagatorPixelSize = self.pixelSize
        self.propagatorDepth = self.depth
      
    def refocus(self, img):
        
        if self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.pixelSize:
            self.updatePropagator(img)
                    
        return refocus(img, self.propagator, background = self.background, window = self.window )
        
    
    def setFindFocusParameters(self, method, roi, margin, depthRange):
        self.findFocusMethod = method
        self.findFocusRoi = roi
        self.findFocusMargin = margin
        self.findFocusDepthRange = depthRange
        
        
    def makePropagatorLUT(self, img, depthRange, nDepths):
        self.propagatorLUT = PropLUT(np.shape(img)[0], self.wavelength, self.pixelSize, depthRange, nDepths)
     
        
    def clearPropagatorLUT(self):
        self.propagatorLUT = None
        
        
    def findFocus(self, img):        
                
        args = {'background': self.background,
                "window": self.window,
                "roi": self.findFocusRoi,
                "margin": self.findFocusMargin,
                "propagatorLUT": self.propagatorLUT}
        
        return findFocus(img, self.wavelength, self.pixelSize, self.findFocusDepthRange, self.findFocusMethod, **args)
    

    def depthStack(self, img, depthRange, nDepths):
        
        args = {'background': self.background,
                "window": self.window}
                
        return refocusStack(img, self.wavelength, self.pixelSize, depthRange, nDepths, **args)

    def offAxisRecon(self, img):
        recon = offAxisDemod(img, self.cropCentre, self.cropRadius)
        if self.relativePhase == True:
            recon = relativePhase(recon, self.backgroundField)
        return recon
    

        

###### Lower-level functions ##########


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
    
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    imgIsFourier = kwargs.get('FourierDomain', False)
    
    
    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if imgIsFourier:  
        
       return np.fft.ifft2(np.fft.fftshift(img * propagator))

    else:   # If we are sent the spatial domain image
        cHologram = preProcess(img, **kwargs)
    
        return np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(cHologram)) * propagator))

   

# Background subtraction and windowing
def preProcess(img, **kwargs):
    
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    #imgIsFourier = kwargs.get('FourierDomain', False)
    
    if background is not None:
        imgOut = img.astype('float32') - background.astype('float32')
    else:
        imgOut  = img.astype('float32')
            
    if window is not None:
        imgOut = imgOut * window
            
    return imgOut

    
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


# Numerical refocusing of a hologram to produce a depth stack. 'depthRange' is a tuple
# defining the min and max depths, the resulting stack will have 'nDepths' images
# equally spaced between these limits. Specify 'imgisFFT' = true if the provided
# 'img' is aready in Fourier domain.
def refocusStack(img, wavelength, pixelSize, depthRange, nDepths, **kwargs):
    cHologram = preProcess(img, **kwargs)
    cHologramFFT = np.fft.fftshift(np.fft.fft2(cHologram))
    depths = np.linspace(depthRange[0], depthRange[1], nDepths)
    kwargs['imgIsFFT'] = True
    imgStack = FocusStack(img, depthRange, nDepths)
    for idx, depth in enumerate(depths):
        prop = propagator(np.shape(img)[0], wavelength, pixelSize, depth)
        imgStack.addIdx(refocus(img, prop, **kwargs), idx)
    return imgStack


# Removes spatial modulation from off axis hologram. cropCentre is the location of
# the modulation frequency in the Fourier Domain, cropRadius is the size of
# the spatial frequency range to keep around the modulation frequency (in FFT pixels)
def offAxisDemod(cameraImage, cropCentre, cropRadius, **kwargs):
    
    
    returnFFT = kwargs.get('returnFFT', False)
    

    # Size of image in pixels (assume square);
    nPoints = np.shape(cameraImage)[0]        
     
    # Make a circular mask
    t1 = time.time()
    [xM, yM] = np.meshgrid(range(nPoints), range(nPoints))
    mask = (xM - cropCentre[0])**2 + (yM - cropCentre[1])**2 < cropRadius**2;
    t2 = time.time()
    #print(round(t2-t1,4))
  
  
    # Apply 2D FFT
    cameraFFT = np.fft.fftshift(np.fft.fft2(cameraImage))
    t3 = time.time()
    #print(round(t3-t2,4))
    
    # Apply the mask
    maskedFFT = cameraFFT * mask;
    t4 = time.time()
    #print(round(t4-t3,4))



    # Shift the ROI to the centre
    shiftedFFT = maskedFFT[round(cropCentre[1] - cropRadius): round(cropCentre[1] + cropRadius),round(cropCentre[0] - cropRadius): round(cropCentre[0] + cropRadius)]
    #t5 = time.time()
    #print(round(t5-t4,4))


    # Reconstruct complex field
    reconField = np.fft.ifft2(np.fft.fftshift(shiftedFFT));
    #t6 = time.time()
    #print(round(t6-t5,4))
    # Remove phase information where amplitude is very low
    #reconThresh = angle(reconField) .* (abs(reconField) > max(abs(reconField(:))) / threshold );
    
   # plt.figure()
   # plt.imshow(np.log(np.abs(maskedFFT)))
   # plt.title('Shifted FFT')
    if returnFFT:
        return reconField, np.log(np.abs(cameraFFT))
    else:
        return reconField

# Finds the location of the off-axis holography modulation peak in the FFT. Finds
# the peak in the positive x region.
def offAxisFindMod(cameraImage):
    
    # Apply 2D FFT
    cameraFFT = np.abs(np.fft.fftshift(np.fft.fft2(cameraImage)) )     

    # Mask central region
    imSize = min(np.shape(cameraImage))
    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    cameraFFT[round(cx - imSize / 8): round(cx + imSize / 8), round(cy - imSize / 8): round(cy + imSize / 8)  ] = 0
    cameraFFT[0: round(cx), :  ] = 0

    peakLoc = np.unravel_index(cameraFFT.argmax(), cameraFFT.shape)
    
    return peakLoc


# Estimates the correct off axis crop radius based on modulation peak position
def offAxisFindCropRadius(cameraImage):
    
    peakLoc = offAxisFindMod(cameraImage)
    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    peakDist = np.sqrt((peakLoc[0] - cx)**2 + (peakLoc[1] - cy)**2)
    
    
    # In the optimal case, the radius is 1/3rd of the modulation position
    cropRadius = math.floor(peakDist / 3)
    
    # Ensure it doesn't run off edge of image
    cropRadius = min (cropRadius, peakLoc[0], np.shape(cameraImage)[0] - peakLoc[0], peakLoc[1], np.shape(cameraImage)[1] - peakLoc[1] )
    
    
    return cropRadius
    

# Predicts the location of the modulation peak (i.e. carrer frequency) in the
# FFT. Returns the distance of the peak from the centre (dc) of the FFT in pixels.
def offAxisPredictMod(wavelength, pixelSize, tiltAngle): 
           
    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength     

     
    # Spatial frequency of mdulation
    refFreq = k * math.sin(tiltAngle)
    
    # Spatial frequency in camera pixels
    refFreqPx = refFreq / pixelSize
    
    # Pixel in Fourier Domain
    modFreqPx = 2 / refFreqPx
    
    return modFreqPx


# Predicts the reference beam tilt based on the modulation of the camera image
# and specified wavelength and pixel size.
def offAxisPredictTiltAngle(cameraImage, wavelength, pixelSize):
    
    
    # Wavenumber
    k = 2 * math.pi / wavelength    

    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    
    # Find the location of the peak
    peakLoc = offAxisFindMod(cameraImage)
    
    hPixelSF = 1 / (2 * pixelSize * np.shape(cameraImage)[0])
    vPixelSF = 1 / (2 * pixelSize * np.shape(cameraImage)[1])

    
    spatialFreq = np.sqrt( (hPixelSF * (peakLoc[0] - cx))**2  + (vPixelSF * (peakLoc[1] - cy) )**2)

   
    tiltAngle = math.asin(spatialFreq / k)
    
    
    return tiltAngle
    


# Remove global phase from using reference (background) images
def relativePhase(img, background):
    
    fieldOut = img * np.exp(1j * -np.angle(background))
    
    return fieldOut


def fourierPlaneDisplay(img):
    
    cameraFFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    return cameraFFT    


# Generates a simple, non-rigorous DIC-style image for display. The image
# should appear similar to a relief map, with dark and light regions
# correspnding to positive and negative phase gradients along the
# shear angle direction (default is horizontal = 0 rad). Phase gradient
# is multiple by the image intensity. 'img' should be a complex numpy array.
def syntheticDIC(img, **kwargs):
    
    shearAngle = kwargs.get('shearAngle', 0)

    
    # Calculate gradient on original image and image phase shifted by pi. Using
    # the smallest phase gradient avoids effects due to phase wrapping
    sobelC1 = phaseGradientAmp(img)
    sobelC2 = phaseGradientAmp(img * np.exp(1j * math.pi))
    
    use1 = np.abs(sobelC1) < np.abs(sobelC2)
    
    sobelC1[np.invert(use1)] = 0
    sobelC2[use1] = 0
    sobelC = sobelC1 + sobelC2
    
    # Rotate the gradient to shear angle
    sobelC = sobelC * np.exp(1j * shearAngle)
       
    # DIC is product of phase gradient along one direction and image intensity
    DIC = np.real(sobelC) * (np.max(np.abs(img)) - np.abs(img))
    
        
    return DIC



def phaseGradientAmp(img):
    
    # Phase gradient in x and y directions
    sobelx = cv.Sobel(np.angle(img),cv.CV_64F,1,0)                  # Find x and y gradients
    sobely = cv.Sobel(np.angle(img),cv.CV_64F,0,1)
    sobelC = sobelx + 1j * sobely

    return sobelC

# Produces a phase gradient (magnitude) image. img should be a complex numpy
# array
def phaseGradient(img):
    
    # Phase gradient in x and y directions
    phaseGradientAmp(img)
    
    phaseGrad1 = np.abs(phaseGradientAmp(img))
    phaseGrad2 = np.abs(phaseGradientAmp(img * np.exp(1j * math.pi)))
    phaseGrad = np.minimum(phaseGrad1, phaseGrad2)
    
    return phaseGrad


# Returns the mean phase in an image
def meanPhase(img):

    meanPhase = np.angle(np.sum(img))
    return meanPhase


# Makes the phase in an image relative to the mean phase in specified ROI
def relativePhaseROI(img, roi):    
    
    avPhase = meanPhase(roi.crop(img))
  
    outImage = img / np.exp(1j * avPhase)
    
    return outImage



############### Utility class for region of interest
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
        return img[self.x: self.x + self.width, self.y:self.y + self.height]
    

################## LUT of propagators
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
        
        
class FocusStack:
     
    def __init__(self, img, depthRange, nDepths):
        self.stack = np.zeros((nDepths, np.shape(img)[0], np.shape(img)[1]), dtype = 'complex128')
        self.depths = np.linspace(depthRange[0], depthRange[1], nDepths)
        self.minDepth = depthRange[0]
        self.maxDepth = depthRange[1]
        self.nDepths = nDepths
        self.depthRange = depthRange
        
    def __str__(self):
        return "Refocus stack. Min: " + str(self.minDepth) + ", Max: " + str(self.maxDepth) + ", Num: " + str(self.nDepths) + ", Step: " + str((self.maxDepth - self.minDepth) / self.nDepths)
        
    def addIdx(self, img, idx):
        self.stack[idx, :,:] = img        
        
    def addDepth(self, img, depth):
        self.stack[self.depthToIndex(depth),:,:] = img
        
    def getIndex(self, idx):
        #print("Getting image at index ", idx)
        #print(self.stack[idx, : , :])
        return self.stack[idx, : , :]
    
    def getDepth(self, depth):
        #print("Getting image from depth ", depth)
        return self.getIndex(self.depthToIndex(depth))
        
    def getDepthIntensity(self, depth):
        #print("Getting image intensity from depth ", depth)
        #print(self.getDepth(depth))
        return np.abs(self.getDepth(depth))
    
    def getIndexIntensity(self, idx):
        #print("Getting image intensity at index ", idx)
        return np.abs(self.getIndex(idx))
    
    def depthToIndex(self, depth):
        idx = round((depth - self.minDepth) / (self.maxDepth - self.minDepth) * self.nDepths)
        if idx < 0:
            idx = 0
        if idx > self.nDepths - 1:
            idx = self.nDepths - 1
        return idx
    
    def writeIntensityToTif(self, filename):
        imlist = []
        for m in self.stack:
            imlist.append(Image.fromarray(255 * np.abs(m).astype('uint16')))

        imlist[0].save(filename, compression="tiff_deflate", save_all=True,
               append_images=imlist[1:])
        
    def writePhaseToTif(self, filename):
        imlist = []
        for m in self.stack:
            im = (np.angle(m) + math.pi) * 255
            imlist.append(Image.fromarray(im.astype('uint16')))

        imlist[0].save(filename, compression="tiff_deflate", save_all=True,
               append_images=imlist[1:])
        
         
# Extracts sqaure of size boxSize from centre of img
def extractCentral(img, boxSize):
       w = np.shape(img)[0]
       h = np.shape(img)[1]

       cx = w/2
       cy = h/2
       boxSemiSize = min(cx,cy,boxSize)
        
       img = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
       return img
   
def get8bit(img):
        
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
    amp = amp / np.max(amp) * 255
    amp = amp.astype('uint8')
    
    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2* math.pi) * 255
    phase = phase.astype('uint8')
    
    return amp, phase