# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

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
from matplotlib import pyplot as plt
import math
import scipy
import scipy.optimize
import time
try:
    import cupy as cp
except:
    pass
    
from PIL import Image
import cv2 as cv

from PyHoloscope.roi import *
from PyHoloscope.focus_stack import *
from PyHoloscope.prop_lut import *

from skimage.restoration import unwrap_phase

def __init__():
    pass


INLINE_MODE = 1
OFFAXIS_MODE = 2

      


def propagator(gridSize, wavelength, pixelSize, depth):
    """ Creates Fourier domain propagator for angular spectrum meethod. GridSize
    is size of image (in pixels) to refocus. 
    """
    
    area = gridSize * pixelSize

    (xM, yM) = np.meshgrid(range(gridSize), range(gridSize))
    
    delta0 = 1/area;
    u = delta0*(xM - gridSize/2 -1);
    v = delta0*(yM - gridSize/2 -1);
    prop= np.exp(1j*math.pi*wavelength*depth*(u**2 + v**2))
    #prop[(alpha**2 + beta**2) > 1] = 0

    
    return prop



def refocus(img, propagator, **kwargs):    
    """ Refocus using angular spectrum method. Takes a hologram (with any pre-processing
    such as background removal already performed) and a pre-computed propagator. 
    """
    
    imgIsFourier = kwargs.get('FourierDomain', False)
    cuda = kwargs.get('cuda', False)
    if np.shape(img) != np.shape(propagator):
        return None
    
    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if imgIsFourier:  
        if cuda is True:
            img2 = cp.array(img)
            propagator2 = cp.array(propagator)
           
            return cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(img2 * propagator2)))
        else:
            return np.fft.ifft2(np.fft.fftshift(img * propagator))

    else:   # If we are sent the spatial domain image
        cHologram = pre_process(img, **kwargs)
        if cuda is True:
            cHologram2 = cp.array(cHologram)
            propagator2 = cp.array(propagator)
            return cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(cp.fft.fftshift(cp.fft.fft2((cHologram2))) * propagator2)))
        else:
            return np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2((cHologram))) * propagator))

   

def pre_process(img, **kwargs):
    """ Carries out steps required prior to refocus - background correction and 
    windowing. Also coverts image to either float32 (if input img is real) or
    complex64 (if input img is complex). Finally, image is cropped to a square
    as non-square images are not currently supported.
    """    

    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    
    if np.iscomplex(img[0,0]):
        imType = 'complex64'
    else:
        imType = 'float32'
                
    
    if background is not None:
        imgOut = img.astype(imType) - background.astype(imType)
    else:
        imgOut  = img.astype(imType)
        
    minSize = np.min(np.shape(imgOut))
    imgOut = imgOut[:minSize, :minSize]
            
    if window is not None:
        if np.iscomplex(img[0,0]):
            imgOut = np.abs(imgOut) * window * np.exp(1j * np.angle(imgOut) * window)
        else:
            imgOut = imgOut * window.astype(imType)
            
    return imgOut


def post_process(img, **kwargs):
    """ Processing after refocus - background subtraction and windowing"""

    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    
    if np.iscomplex(img[0,0]):
        imType = 'complex64'
    else:
        imType = 'float32'
    
    if background is not None:
        imgOut = img.astype(imType) - background.astype(imType)
    else:
        imgOut  = img.astype(imType)
            
    if window is not None:
        imgOut = imgOut * window
            
    return imgOut

    
def circ_cosine_window(imgSize, circleRadius, skinThickness):
    """ Produce a circular cosine window mask on grid of imgSize * imgSize. Mask
    is 0 for radius > circleSize and 1 for radius < (circleSize - skinThickness)
    The intermediate region is a smooth cosine function.
    """
    
    innerRad = circleRadius - skinThickness
    xM, yM = np.meshgrid(range(imgSize),range(imgSize))
    imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
    mask =  np.cos(math.pi / (2 * skinThickness) * (imgRad - innerRad))**2
    mask[imgRad < innerRad ] = 1
    mask[imgRad > innerRad + skinThickness] = 0
    return mask

def square_cosine_window(imgSize, circleRadius, skinThickness):
    """ TODO
    """
    return circ_cosine_window(imgSize, circleRadius, skinThickness)



def focus_score(img, method, **kwargs):
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
    
    cHologram = pre_process(img, background=background, window = window)
    
    # If a margin is specified, this means we only refocus the ROI plus a 
    # margin around it for speed. Define the ROI here.
    if margin is not None and scoreROI is not None:
        refocusROI = roi(scoreROI.x - margin, scoreROI.y - margin, scoreROI.width + margin *2, scoreROI.height + margin *2)
        refocusROI.constrain(0,0,np.shape(img)[0], np.shape(img)[1])
        scoreROI = roi(margin, margin, scoreROI.width, scoreROI.height)
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
    print(startDepth)
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
        score.append(refocus_and_score(depth, cHologramFFT, pixelSize, wavelength, method, scoreROI,  None))
        
    return score, depths


def refocus_stack(img, wavelength, pixelSize, depthRange, nDepths, **kwargs):

    """ Numerical refocusing of a hologram to produce a depth stack. 'depthRange' is a tuple
    defining the min and max depths, the resulting stack will have 'nDepths' images
    equally spaced between these limits. Specify 'imgisFFT' = true if the provided
    'img' is aready in Fourier domain.   
    """
    window = kwargs.get('window', None)
    cHologram = pre_process(img, **kwargs)
    cHologramFFT = np.fft.fftshift(np.fft.fft2(cHologram))
    depths = np.linspace(depthRange[0], depthRange[1], nDepths)
    kwargs['imgIsFFT'] = True
    imgStack = FocusStack(cHologram, depthRange, nDepths)

    for idx, depth in enumerate(depths):
        prop = propagator(np.shape(img)[0], wavelength, pixelSize, depth)
        imgStack.addIdx(post_process(refocus(cHologramFFT, prop, **kwargs), window=window), idx)
   
    return imgStack


def off_axis_demod(cameraImage, cropCentre, cropRadius, **kwargs):
    """ Removes spatial modulation from off axis hologram. cropCentre is the location of
    the modulation frequency in the Fourier Domain, cropRadius is the size of
    the spatial frequency range to keep around the modulation frequency (in FFT pixels)    
    """
    
    returnFFT = kwargs.get('returnFFT', False)
    mask = kwargs.get('mask', None)
    cuda = kwargs.get('cuda', False)
    
    #cameraImage = cameraImage[0:]

    # Size of image in pixels (assume square);
    nPoints = np.min(np.shape(cameraImage))
    cameraImage = cameraImage[0:nPoints, 0:nPoints]       
     
    # Make a circular mask
    if mask is None:
        [xM, yM] = np.meshgrid(range(cropRadius * 2), range(cropRadius *2))
        mask = (xM - cropRadius)**2 + (yM - cropRadius)**2 < cropRadius**2
        mask = mask.astype('complex')
  
    # Apply 2D FFT
    if cuda is False:
        cameraFFT = np.fft.fftshift(np.fft.fft2(cameraImage))
    else:
        cameraFFT = cp.fft.fftshift(cp.fft.fft2(cp.array(cameraImage)))    
   
    # Shift the ROI to the centre
    shiftedFFT = cameraFFT[round(cropCentre[1] - cropRadius): round(cropCentre[1] + cropRadius),round(cropCentre[0] - cropRadius): round(cropCentre[0] + cropRadius)]

    # Apply the mask
    if cuda is True:
        mask = cp.array(mask)
    maskedFFT = shiftedFFT * mask

    # Reconstruct complex field
    if cuda is False:
        reconField = np.fft.ifft2(np.fft.fftshift(shiftedFFT))
    else:
        reconField = cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(shiftedFFT)))
   
    if returnFFT:
        if cuda is True:
            try:
                cameraFFT = cp.asnumpy(cameraFFT)
            except:
                pass
        return reconField, np.log(np.abs(cameraFFT) + 0.000001)
    
    else:
        return reconField
    

def off_axis_find_mod(cameraImage):
    """ Finds the location of the off-axis holography modulation peak in the FFT. Finds
    the peak in the positive x region.    
    """
    
    # Apply 2D FFT
    cameraFFT = np.transpose(np.abs(np.fft.fftshift(np.fft.fft2(cameraImage)) ) )  
 
    # Mask central region
    imSize = min(np.shape(cameraImage))
    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    cameraFFT[round(cy - imSize / 8): round(cy + imSize / 8), round(cx - imSize / 8): round(cx + imSize / 8)  ] = 0
    cameraFFT[round(cy):, :  ] = 0
 
    peakLoc = np.unravel_index(cameraFFT.argmax(), cameraFFT.shape)
    
    return peakLoc


def off_axis_find_crop_radius(cameraImage):
    """ Estimates the correct off axis crop radius based on modulation peak position
    """
    peakLoc = off_axis_find_mod(cameraImage)
    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    peakDist = np.sqrt((peakLoc[0] - cx)**2 + (peakLoc[1] - cy)**2)
    
    # In the optimal case, the radius is 1/3rd of the modulation position
    cropRadius = math.floor(peakDist / 3)
    
    # Ensure it doesn't run off edge of image
    cropRadius = min (cropRadius, peakLoc[0], np.shape(cameraImage)[0] - peakLoc[0], peakLoc[1], np.shape(cameraImage)[1] - peakLoc[1] )
    
    return cropRadius


def off_axis_predict_mod(wavelength, pixelSize, tiltAngle): 
    """ Predicts the location of the modulation peak (i.e. carrer frequency) in the
    FFT. Returns the distance of the peak from the centre (dc) of the FFT in pixels.
    """
           
    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength     
     
    # Spatial frequency of mdulation
    refFreq = k * math.sin(tiltAngle)
    
    # Spatial frequency in camera pixels
    refFreqPx = refFreq / pixelSize
    
    # Pixel in Fourier Domain
    modFreqPx = 2 / refFreqPx
    
    return modFreqPx


def off_axis_predict_tilt_angle(cameraImage, wavelength, pixelSize):
    """ Predicts the reference beam tilt based on the modulation of the camera image
    and specified wavelength and pixel size.
    """    
    
    # Wavenumber
    k = 2 * math.pi / wavelength    

    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    
    # Find the location of the peak
    peakLoc = off_axis_find_mod(cameraImage)
    
    hPixelSF = 1 / (2 * pixelSize * np.shape(cameraImage)[0])
    vPixelSF = 1 / (2 * pixelSize * np.shape(cameraImage)[1])
    
    spatialFreq = np.sqrt( (hPixelSF * (peakLoc[0] - cx))**2  + (vPixelSF * (peakLoc[1] - cy) )**2)
   
    tiltAngle = math.asin(spatialFreq / k)
    
    return tiltAngle
    

def relative_phase(img, background):
    """ Remove global phase from complex image using reference (background) field 
    """    
    
    if np.iscomplexobj(img):
        phase = np.angle(img)
    else:
        phase = img
        
    if np.iscomplexobj(background):
        backgroundPhase = np.angle(background)
    else:
        backgroundPhase = background
        
    phaseOut = phase - backgroundPhase    
    
    if np.iscomplexobj(img):             
        return np.abs(img) * np.exp(1j * phaseOut)
    else:     
        return phaseOut
        

def stable_phase(img, roi = None):
    """ Subtracts the mean phase from the phase map, removing global phase
    fluctuations. Can accept complex img, as a field, or a real img, which
    is unwrapped phase in radians 
    """
   
    if roi is not None:
        imgCrop = roi.crop(img)
    else:
        imgCrop = img 
   
    if np.iscomplexobj(img):
        phase = np.angle(img)
        phaseCrop = np.angle(imgCrop)
    else:
        phase = img
        phaseCrop = imgCrop   
    
        
    avPhase = mean_phase(imgCrop)
    #avPhase = np.mean(imgCrop)
    #print(f"general_stable_phase_ROI, {roi}, Average phase b4: {avPhase}")

    phaseOut = phase - avPhase

    #avPhase = mean_phase(phaseOut)
    #avPhase = np.mean(phaseOut)
    #print(f"general_stable_phase_ROI, {roi}, Average phase af: {mean_phase(phaseOut)}")


    if np.iscomplexobj(img):             
        return np.abs(img) * np.exp(1j * phaseOut)
    else:     
        return phaseOut


def obtain_tilt(img):
    """ Estimates the global tilt in the 2D unwrapped phase (e.g. caused by tilt in coverglass). img
    should be unwrapped phase (real)
    """
    
    tiltX, tiltY = np.gradient(img)
    tiltX = np.mean(tiltX)
    tiltY = np.mean(tiltY)
    
    mx, my = np.indices(np.shape(img))
    
    tilt = mx * tiltX + my * tiltY
   
    return tilt
 
    
def phase_unwrap(img):
    """ 2D phase unwrapping. img should be wrapped phase (real)
    """    
    img = unwrap_phase(img)

    return img


def fourier_plane_display(img):
    """ Return a log-scale Fourier plane for display
    """
    cameraFFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    return cameraFFT    


def synthetic_DIC(img, **kwargs):

    """ Generates a simple, non-rigorous DIC-style image for display. The image
    should appear similar to a relief map, with dark and light regions
    correspnding to positive and negative phase gradients along the
    shear angle direction (default is horizontal = 0 rad). Phase gradient
    is multiplied by the image intensity. 'img' should be a complex numpy array.    
    """
    
    shearAngle = kwargs.get('shearAngle', 0)
    
    # Calculate gradient on original image and image phase shifted by pi. Using
    # the smallest phase gradient avoids effects due to phase wrapping
    sobelC1 = phase_gradient_amp(img)
    sobelC2 = phase_gradient_amp(img * np.exp(1j * math.pi))
    
    use1 = np.abs(sobelC1) < np.abs(sobelC2)
    
    sobelC1[np.invert(use1)] = 0
    sobelC2[use1] = 0
    sobelC = sobelC1 + sobelC2
    # Rotate the gradient to shear angle
    sobelC = sobelC * np.exp(1j * shearAngle)
       
    # DIC is product of phase gradient along one direction and image intensity
    DIC = np.real(sobelC) * (np.max(np.abs(img)) - np.abs(img)) 
    # Not sure how best to involvw amplitude here
    # DIC = np.real(sobelC) * (-np.abs(img))
        
    return DIC


def phase_gradient_amp(img):
    """ Returns the ampitude of the phase gradient
    """
    
    # Phase gradient in x and y directions
    sobelx = cv.Sobel(np.angle(img),cv.CV_64F,1,0)                  # Find x and y gradients
    sobely = cv.Sobel(np.angle(img),cv.CV_64F,0,1)
    sobelC = sobelx + 1j * sobely

    return sobelC


def phase_gradient(img):
    """ Produces a phase gradient (magnitude) image. img should be a complex numpy
    array
    """
    
    # Phase gradient in x and y directions
    phaseGrad1 = np.abs(phase_gradient_amp(img))
    phaseGrad2 = np.abs(phase_gradient_amp(img * np.exp(1j * math.pi)))
   
    phaseGrad = np.minimum(phaseGrad1, phaseGrad2)
    
    return phaseGrad


def mean_phase(img):
    """Returns the mean phase in a complex field
    """
    if np.iscomplexobj(img):
        meanPhase = np.angle(np.sum(img))
    else:
        meanPhase = np.mean(img)
    return meanPhase


def relative_phase_ROI(img, roi):    
    """ Makes the phase in an image relative to the mean phase in specified ROI
    """
    avPhase = mean_phase(roi.crop(img))
    outImage = img / np.exp(1j * avPhase)
    
    return outImage

        
# Extracts sqaure of size boxSize from centre of img
def extract_central(img, boxSize):
       """ Extracts sqaure of size boxSize from centre of img
       """
       w = np.shape(img)[0]
       h = np.shape(img)[1]

       cx = w/2
       cy = h/2
       boxSemiSize = min(cx,cy,boxSize)
        
       img = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
       return img
   
   
def amplitude(img):
    """ Returns amplitude of complex image
    """
    return np.abs(img)
   
     
def phase(img):
    """ Returns phase of complex image
    """ 
    return np.angle(img) % (2 * math.pi)  
    

def get8bit(img):
    """ Takes complex image and returns 8 bit representations of the amplitude and
    phase for saving as 8 bit images          
    """
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
    amp = amp / np.max(amp) * 255
    amp = amp.astype('uint8')
    
    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2* math.pi) * 255
    phase = phase.astype('uint8')
    
    return amp, phase


def get16bit(img):
    """ Takes complex image and returns 16 bit representations of the amplitude and
    phase for saving as 8 bit images           
    """
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
    amp = amp / np.max(amp) * 255
    amp = amp.astype('uint16')
    
    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2* math.pi) * 255
    phase = phase.astype('uint16')
    
    return amp, phase


def save_phase_image(img, filename):
    """ Saves phase information as 16 bit tiff scaled so that 2pi = 256
    """
    
    if np.iscomplexobj(img):
        phase = np.angle(img).astype('double')
    else:
        phase = img.astype('double')
    phase = phase - np.min(phase)    
    phase = ((phase / (2 * math.pi)) * 256).astype('uint16')

    im = Image.fromarray(phase)
    im.save(filename)
        
        
