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

      

# Creates Fourier domain propagator for angular spectrum meethod. GridSize
# is size of image (in pixels) to refocus.
def propagator(gridSize, wavelength, pixelSize, depth):
    
    
    area = gridSize * pixelSize

    (xM, yM) = np.meshgrid(range(gridSize), range(gridSize))
    
   
    
    
    delta0 = 1/area;
    u = delta0*(xM - gridSize/2 -1);
    v = delta0*(yM - gridSize/2 -1);
    prop= np.exp(1j*math.pi*wavelength*depth*(u**2 + v**2))
    #prop[(alpha**2 + beta**2) > 1] = 0

    
    return prop




# Refocus using angular spectrum method. Takes a hologram (with any pre-processing
# such as background removal already performed) and a pre-computed propagator. 
def refocus(img, propagator, **kwargs):
    
   
    imgIsFourier = kwargs.get('FourierDomain', False)
    if np.shape(img) != np.shape(propagator):
        #print("shapre mismatch")
        return None
    
    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if imgIsFourier:  
        
       return np.fft.ifft2(np.fft.fftshift(img * propagator))

    else:   # If we are sent the spatial domain image
        cHologram = pre_process(img, **kwargs)
    
        return np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2((cHologram))) * propagator))

   

# Processing prior to refocus - background subtraction and windowing
def pre_process(img, **kwargs):
    
    background = kwargs.get('background', None)
    window = kwargs.get('window', None)
    #imgIsFourier = kwargs.get('FourierDomain', False)
    
    if np.iscomplex(img[0,0]):
        imType = 'complex64'
    else:
        imType = 'float32'
                
    
    if background is not None:
        imgOut = img.astype(imType) - background.astype(imType)
    else:
        imgOut  = img.astype(imType)
            
    if window is not None:
        if np.iscomplex(img[0,0]):
            imgOut = np.abs(imgOut) * window * np.exp(1j * np.angle(imgOut) * window)
        else:
            imgOut = imgOut * window.astype(imType)
            
    return imgOut




# Processing after refocus - background subtraction and windowing
def post_process(img, **kwargs):
    
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




    
# Produce a circular cosine window mask on grid of imgSize * imgSize. Mask
# is 0 for radius > circleSize and 1 for radius < (circleSize - skinThickness)
# The intermediate region is a smooth cosine function.
def circ_cosine_window(imgSize, circleRadius, skinThickness):
    innerRad = circleRadius - skinThickness
    xM, yM = np.meshgrid(range(imgSize),range(imgSize))
    imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
    mask =  np.cos(math.pi / (2 * skinThickness) * (imgRad - innerRad))**2
    mask[imgRad < innerRad ] = 1
    mask[imgRad > innerRad + skinThickness] = 0
    return mask

# TODO
def square_cosine_window(imgSize, circleRadius, skinThickness):
    
    return circ_cosine_window(imgSize, circleRadius, skinThickness)



# Returns score of how 'in focus' an image is based on selected method.
# Brenner, Sobel or Peak
def focus_score(img, method, **kwargs):
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
        
        return -(stDev[0,0]**2)
        
        
        
    return focusScore





# Refocuses an image to specificed depth and returns focus score, used by
# findFocus
def refocus_and_score(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT):
    
    t1 = time.perf_counter()
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
    
    
    score = focusScore(refocImg, method)
    print(depth, score)
    #print(time.perf_counter() - t1)
    return score



# Determine optimal depth to maximise focus metric in image
def find_focus(img, wavelength, pixelSize, depthRange, method, **kwargs):
        
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
    depth =  scipy.optimize.minimize_scalar(refocusAndScore, method = 'bounded', bounds = depthRange, args= (imgFFT ,pixelSize, wavelength, method, scoreROI, propLUT) )

    return depth.x 








# An initial check for approximate location of good focus depths prior to a finer serrch. Called
# by findFocus
def coarse_focus_search(imgFFT, depthRange, nIntervals, pixelSize, wavelength, method, scoreROI, propLUT):
    
    searchDepths = np.linspace(depthRange[0], depthRange[1], nIntervals)
    focusScore = np.zeros_like(searchDepths)
    for idx, depth in enumerate(searchDepths):
        focusScore[idx] = refocus_and_score(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT)
    
    bestInterval = np.argmin(focusScore)
    bestDepth = searchDepths[bestInterval]
    return bestDepth
    
    
    
    
    

# Produce a plot of focus score against depth, mainly useful for debugging
# erroneous focusing
def focus_score_curve(img, wavelength, pixelSize, depthRange, nPoints, method, **kwargs):
        
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



# Numerical refocusing of a hologram to produce a depth stack. 'depthRange' is a tuple
# defining the min and max depths, the resulting stack will have 'nDepths' images
# equally spaced between these limits. Specify 'imgisFFT' = true if the provided
# 'img' is aready in Fourier domain.
def refocus_stack(img, wavelength, pixelSize, depthRange, nDepths, **kwargs):
    window = kwargs.get('window', None)
    cHologram = pre_process(img, **kwargs)
    cHologramFFT = np.fft.fftshift(np.fft.fft2(cHologram))
    depths = np.linspace(depthRange[0], depthRange[1], nDepths)
    kwargs['imgIsFFT'] = True
    #img = pre_process(img, **kwargs)
    imgStack = FocusStack(cHologram, depthRange, nDepths)

    for idx, depth in enumerate(depths):
        prop = propagator(np.shape(img)[0], wavelength, pixelSize, depth)
        imgStack.addIdx(post_process(refocus(img, prop, **kwargs), window=window), idx)
    return imgStack




# Removes spatial modulation from off axis hologram. cropCentre is the location of
# the modulation frequency in the Fourier Domain, cropRadius is the size of
# the spatial frequency range to keep around the modulation frequency (in FFT pixels)
def off_axis_demod(cameraImage, cropCentre, cropRadius, **kwargs):
    
    
    returnFFT = kwargs.get('returnFFT', False)
    mask = kwargs.get('mask', None)
    cuda = kwargs.get('cuda', False)

    
    #cameraImage = cameraImage[0:]

    # Size of image in pixels (assume square);
    nPoints = np.min(np.shape(cameraImage))
    cameraImage = cameraImage[0:nPoints, 0:nPoints]
       
     
    # Make a circular mask
    if mask is None:
        #t1 = time.time()
        [xM, yM] = np.meshgrid(range(cropRadius * 2), range(cropRadius *2))
        mask = (xM - cropRadius)**2 + (yM - cropRadius)**2 < cropRadius**2
        mask = mask.astype('complex')
        #t2 = time.time()
        #print("generate mask", round(t2-t1,4))
  
  
    # Apply 2D FFT
    #t2 = time.time()
    #print(cameraImage.dtype)
    if cuda is False:
        cameraFFT = np.fft.fftshift(np.fft.fft2(cameraImage))
    else:
        cameraFFT = cp.fft.fftshift(cp.fft.fft2(cp.array(cameraImage)))
    #t3 = time.time()
    #print("fft", round(t3-t2,4))
    
   
    # Shift the ROI to the centre
    shiftedFFT = cameraFFT[round(cropCentre[1] - cropRadius): round(cropCentre[1] + cropRadius),round(cropCentre[0] - cropRadius): round(cropCentre[0] + cropRadius)]
    #t4 = time.time()
    #print("crop", round(t4-t3,4))

    # Apply the mask
    #maskedFFT = cameraFFT * mask
    if cuda is True:
        mask = cp.array(mask)
    maskedFFT = shiftedFFT * mask
    #t5 = time.time()
    #print("apply mask", round(t5-t4,4))


    # Reconstruct complex field
    if cuda is False:
        reconField = np.fft.ifft2(np.fft.fftshift(shiftedFFT))
    else:
        reconField = cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(shiftedFFT)))

    #t6 = time.time()
    #print("inverse FFT", round(t6-t5,4))
    # Remove phase information where amplitude is very low
    #reconThresh = angle(reconField) .* (abs(reconField) > max(abs(reconField(:))) / threshold );
    
    # plt.figure()
    # plt.imshow(np.log(np.abs(maskedFFT)))
    # plt.title('Shifted FFT')
   
    if returnFFT:
        if cuda is True:
            try:
                cameraFFT = cp.asnumpy(cameraFFT)
            except:
                pass
        return reconField, np.log(np.abs(cameraFFT) + 0.000001)
    
    else:
        return reconField
    
    
    
    
    
    

# Finds the location of the off-axis holography modulation peak in the FFT. Finds
# the peak in the positive x region.
def off_axis_find_mod(cameraImage):
    
    # Apply 2D FFT
    cameraFFT = np.transpose(np.abs(np.fft.fftshift(np.fft.fft2(cameraImage)) ) )  
 

    # Mask central region
    imSize = min(np.shape(cameraImage))
    cx = np.shape(cameraImage)[0] / 2
    cy = np.shape(cameraImage)[1] / 2
    cameraFFT[round(cy - imSize / 8): round(cy + imSize / 8), round(cx - imSize / 8): round(cx + imSize / 8)  ] = 0
    cameraFFT[round(cy):, :  ] = 0
    #plt.imshow(np.log(cameraFFT + 0.001))

    peakLoc = np.unravel_index(cameraFFT.argmax(), cameraFFT.shape)
    #print(peakLoc)
    
    return peakLoc








# Estimates the correct off axis crop radius based on modulation peak position
def off_axis_find_crop_radius(cameraImage):
    
    peakLoc = off_axis_find_mod(cameraImage)
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
def off_axis_predict_mod(wavelength, pixelSize, tiltAngle): 
           
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
def off_axis_predict_tilt_angle(cameraImage, wavelength, pixelSize):
    
    
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
    



# Remove global phase from complex image using reference (background) field 
def relative_phase(img, background):
    
    
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
     
        
# Subtracts the mean phase from the phase map, removing global phase
# fluctuations. Can accept complex img, as a field, or a real img, which
# is unwrapped phase in radians
def stable_phase(img):
    
    if np.iscomplexobj(img):
        phase = np.angle(img)
        t1 = time.perf_counter()
        av = np.sum(img)
        #print(time.perf_counter() - t1)
        avPhase = np.angle(av)
    else:
        phase = img
        #field = np.ones_like(img) * np.exp(1j * img)
        #av = np.sum(field)
        avPhase = np.mean(phase)
    phaseOut = phase - avPhase

    if np.iscomplexobj(img):             
        return np.abs(img) * np.exp(1j * phaseOut)
    else:     
        return phaseOut


# Estimates the global tilt in the 2D unwrapped phase (e.g. caused by tilt in coverglass). img
# should be unwrapped phase (real)
def obtain_tilt(img):
    
    tiltX, tiltY = np.gradient(img)
    tiltX = np.mean(tiltX)
    tiltY = np.mean(tiltY)
    
    mx, my = np.indices(np.shape(img))
    
    tilt = mx * tiltX + my * tiltY
   
    return tilt
 
    
# 2D phase unwrapping. img should be wrapped phase (real)
def phase_unwrap(img):
    
    img = unwrap_phase(img)

    return img


    

# Return a log-scale Fourier plane for display
def fourier_plane_display(img):
    
    cameraFFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)) ) )
    return cameraFFT    



# Generates a simple, non-rigorous DIC-style image for display. The image
# should appear similar to a relief map, with dark and light regions
# correspnding to positive and negative phase gradients along the
# shear angle direction (default is horizontal = 0 rad). Phase gradient
# is multiplied by the image intensity. 'img' should be a complex numpy array.
def synthetic_DIC(img, **kwargs):
    
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
    # Not sure how best to invovel amplitude here
    # DIC = np.real(sobelC) * (-np.abs(img))

        
    return DIC







# Returns the ampitude of the phase gradient
def phase_gradient_amp(img):
    
    # Phase gradient in x and y directions
    sobelx = cv.Sobel(np.angle(img),cv.CV_64F,1,0)                  # Find x and y gradients
    sobely = cv.Sobel(np.angle(img),cv.CV_64F,0,1)
    sobelC = sobelx + 1j * sobely

    return sobelC







# Produces a phase gradient (magnitude) image. img should be a complex numpy
# array
def phase_gradient(img):
    
    # Phase gradient in x and y directions
   
    phaseGrad1 = np.abs(phase_gradient_amp(img))
    phaseGrad2 = np.abs(phase_gradient_amp(img * np.exp(1j * math.pi)))
    phaseGrad = np.minimum(phaseGrad1, phaseGrad2)
    
    return phaseGrad





# Returns the mean phase in a complex field
def mean_phase(img):
    if np.iscomplexobj(img):
        meanPhase = np.angle(np.sum(img))
    else:
        meanPhase = np.angle(np.sum(exp(1j * img)))
    return meanPhase






# Makes the phase in an image relative to the mean phase in specified ROI
def relative_phase_ROI(img, roi):    
    
    avPhase = mean_phase(roi.crop(img))
  
    outImage = img / np.exp(1j * avPhase)
    
    return outImage





        
# Extracts sqaure of size boxSize from centre of img
def extract_central(img, boxSize):
       w = np.shape(img)[0]
       h = np.shape(img)[1]

       cx = w/2
       cy = h/2
       boxSemiSize = min(cx,cy,boxSize)
        
       img = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
       return img
   
   
def amplitude(img):
    return np.abs(img)
   
    
def phase(img):
    return np.angle(img) % (2 * math.pi)  
    

    
    
# Takes complex image and returns 8 bit representations of the amplitude and
# phase for saving as 8 bit images   
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

# Takes complex image and returns 16 bit representations of the amplitude and
# phase for saving as 8 bit images   
def get16bit(img):
        
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
    amp = amp / np.max(amp) * 255
    amp = amp.astype('uint16')
    
    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2* math.pi) * 255
    phase = phase.astype('uint16')
    
    return amp, phase


       
        
        
