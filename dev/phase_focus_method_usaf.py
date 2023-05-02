# -*- coding: utf-8 -*-
"""
Tests generation of plot of focus score against refocus depth.

@author: Mike Hughes
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')


from matplotlib import pyplot as plt
import numpy as np
import random
import time

import cv2 as cv

import sys
import os

sys.path.append("..\\..\\Pybundle\\src")
import pybundle as pyb

sys.path.append("..\src")
import pyholoscope as pyh
from pyholoscope.roi import Roi

import scipy.signal

hologram = pyh.load_image(r"..\\test\\test data\\inline_example_holo.tif")
background = pyh.load_image(r"..\\test\\test data\\inline_example_back.tif")

hologram = pyh.load_image(r"C:\Users\AOG\OneDrive - University of Kent\Experimental\Holography\Inline Holography of Sample 11_01_22\preprocess\Dataset10 Lillium Anther\im2.tif")
background = None

folder = 'paramecium'

# Bundle pattern removal parameters
imgSize = 400            # Image size after fibre core removal
rad = imgSize/2          # Image radius after fibre core removal
coreSize = 3             # Used to help the core-finding routine




files = [f.path for f in os.scandir(folder)]
holoReconTri = np.zeros((imgSize, imgSize, len(files)))
backgroundFile = None
    
# Search for background file
for f in files:
    if f.rfind('background') > 0:
        backgroundFile = f
   
if backgroundFile is not None:

    # Load background images and use it as a calibration
    background = cv.imread(backgroundFile)[:,:,0]
    fibreCalib = pyb.calib_tri_interp(background, coreSize, imgSize, background=background, normalise = None, autoMask = True)
   
for idx,hologramFile in enumerate(files):
    if hologramFile.rfind('background') < 0:
        hologram = cv.imread(hologramFile)[:,:,0]
        holoReconTri[:,:,idx] = pyb.recon_tri_interp(hologram, fibreCalib)
        



wavelength = 450e-9
pixelSize = 1.5e-6
depthRange = (0, 0.004)
nDepths = 20
offset = 20
size = 40



holo = pyh.Holo(pyh.INLINE_MODE, wavelength, pixelSize)

trialDepths = np.linspace(depthRange[0], depthRange[1], 40)
midPoint = np.shape(holoReconTri)[0] / 2




shiftx = np.zeros(len(trialDepths))
shifty = np.zeros(len(trialDepths))
maxVal = np.zeros(len(trialDepths))

hologram =  holoReconTri[:,:,3]

for fileIdx in range(len(files)):
    hologram =  holoReconTri[:,:,fileIdx]
    
    #holo.set_depth(0.001)
    #refocusImg = holo.process(hologram)
    #holo.set_depth(-0.001)
    #complexImg = holo.process((refocusImg))

    for idx, depth in enumerate(trialDepths):        
        
        holo.set_depth(depth)
        procImg = holo.process(( hologram))
        
        fourierPlane = np.fft.fftshift(np.fft.fft2(procImg))
        
        leftRoi = Roi(midPoint - offset - size / 2, midPoint - size / 2,size,size )
        rightRoi = Roi(midPoint + offset - size / 2 + 1, midPoint - size / 2, size,size)
        
        leftBlock = leftRoi.clear_outside(fourierPlane)
        rightBlock = rightRoi.clear_outside(fourierPlane)
        
        
        #plt.imshow(np.abs(fourierPlane), cmap = 'gray', vmax = 100000)
        
        #plt.figure()
        #plt.imshow((np.abs(leftBlock)), cmap = 'gray')
        
        #plt.figure()
        #plt.imshow(np.abs(rightBlock), cmap = 'gray')
        
        leftFFT = (np.fft.fft2(np.fft.fftshift(leftBlock)))
        rightFFT = (np.fft.fft2(np.fft.fftshift(rightBlock)))
        
        
        leftFFT = np.abs(leftFFT)
        rightFFT = np.abs(rightFFT)

        reconFFT = leftFFT + rightFFT
    
    
        l =  pyb.extract_central(leftFFT, 100).ravel()
        r =  pyb.extract_central(rightFFT, 100).ravel()
        
        l = (l - np.mean(l)) / (np.std(l) * len(l))
        r = (r - np.mean(r)) / (np.std(r))
        maxVal[idx] = np.abs(np.correlate(l, r))

        
        #leftFFT = cv.blur(leftFFT, (5,5))
        #rightFFT = cv.blur(rightFFT, (5,5))
        
        
        # fig, axs = plt.subplots(2,2 ,dpi=150)
        # fig.suptitle(depth)
        # axs[0,0].imshow(np.abs(leftFFT), cmap='gray')
        # axs[0,1].imshow(np.abs(rightFFT), cmap='gray')
        # axs[1,0].imshow(np.abs(reconFFT), cmap='gray')
        # axs[1,1].imshow(np.abs(procImg), cmap='gray')
    
        
    
        shifts = pyb.SuperRes.find_shift(leftFFT, rightFFT, 100, 200, 1)
    
        #cc = scipy.signal.correlate2d(leftFFT, rightFFT)
        # templateSize = 25
        # refSize = 50
        
        # template = pyb.extract_central(rightFFT, templateSize).astype('float32')
        # refIm = pyb.extract_central(leftFFT, refSize).astype('float32')

        
        # cc = cv.matchTemplate(template, refIm, cv.TM_CCORR_NORMED)
        
        # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(cc)
        # shift = [max_loc[0] - (refSize - templateSize) ,
        #          max_loc[1] - (refSize - templateSize)]
        # print(depth, ":", shift) 

        # plt.figure()
        # plt.imshow(cc)
        # plt.title(files[fileIdx] + ":" + str(depth))
      
        shiftx[idx], shifty[idx] = shifts
        
    plt.figure()
    plt.plot(trialDepths * 1000, (shiftx))
    plt.xlabel('Depth (mm)')
    plt.ylabel('Shift')
    plt.title(files[fileIdx] +": X Shift")
    
    
    plt.figure()
    plt.plot(trialDepths * 1000, (maxVal))
    plt.xlabel('Depth (mm)')
    plt.ylabel('Correlation')
    plt.title(files[fileIdx] + ": Max")




# Test found focus
# foundDepth = 0.00105 # trialDepths[np.argmax(shiftx)]
# holo.set_depth(foundDepth)
# focusImg = np.abs(holo.process(hologram))

# plt.figure(dpi=150)
# plt.imshow(focusImg, cmap='gray')
# plt.title("Autofocused Image")




# depths = np.linspace(depthRange[0], depthRange[1], nDepths)

# focusMetrics = ['Brenner', 'Peak', 'Sobel', 'SobelVariance', 'Var', 'DarkFocus']

# roi = pyh.Roi(390,280,250,250)

# holo = pyh.Holo(pyh.INLINE_MODE, wavelength, pixelSize)
# holo.set_background(background)
holo.set_depth(depth)
# refocusImg = holo.process(hologram)

# refocsStack = holo.depth_stack(hologram, depthRange, nDepths)
# refocsStack.write_intensity_to_tif('stack.tif')

# plt.figure()
# plt.imshow(pyh.amplitude(refocusImg), cmap='gray')


# plt.figure()
# plt.xlabel('Depth (mm)')
# plt.ylabel('Focus Score')


# focusCurve = np.zeros((len(focusMetrics),len(depths)))
# for metricIdx, focusMetric in enumerate(focusMetrics):
#     for idx, depth in enumerate(depths):   
    
    
#         cImg = pyh.amplitude(refocsStack.get_index(idx))
#         cImg = roi.crop(cImg)
#         focusCurve[metricIdx, idx] = pyh.focus_score(cImg, focusMetric)
    
    
#     focusCurveNorm = focusCurve[metricIdx,:] - np.min(focusCurve[metricIdx,:])
#     focusCurveNorm = focusCurveNorm / np.max(focusCurveNorm)
#     plt.plot(depths * 1000, focusCurveNorm, label = focusMetric)

# plt.legend()


# # for idx, depth in enumerate(depths):   


# #     cImg = pyh.amplitude(refocsStack.get_index(idx))
# #     cImg = roi.crop(cImg)
# #     focusCurve[idx] = pyh.focus_score(cImg, 'DarkFocus')


# #focusCurve = focusCurve - np.min(focusCurve)
# #focusCurve = focusCurve / np.max(focusCurve)
# #plt.plot(depths * 1000, focusCurve, label = focusMetric)


# holo.set_use_numba(True)
# t1 = time.perf_counter()
# depth = holo.auto_focus(hologram, depthRange = depthRange, method = 'DarkFocus', roi = roi, margin = None)
# print("Refocus time", time.perf_counter() - t1)

# holo.set_depth(depth)
# refocusImg = holo.process(hologram)
# plt.imshow(pyh.amplitude(refocusImg), cmap='gray')

# # depth = 0.00035
# # prop = pyh.propagator(imgSize, wavelength, pixelSize, depth)
# # refocus = pyh.refocus(holoProc - backgroundProc, prop)
# # plt.figure()
# # plt.imshow(pyh.amplitude(refocus), cmap='gray')