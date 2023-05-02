# -*- coding: utf-8 -*-
"""
Tests generation of plot of focus score against refocus depth.

@author: Mike Hughes
"""

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

hologram = pyh.load_image(r"..\\test\\test data\\inline_example_holo.tif")
background = pyh.load_image(r"..\\test\\test data\\inline_example_back.tif")

hologram = pyh.load_image(r"C:\Users\AOG\OneDrive - University of Kent\Experimental\Holography\Inline Holography of Sample 11_01_22\preprocess\Dataset10 Lillium Anther\im2.tif")
background = None

# Holography
#wavelength = 630e-9
#pixelSize = 1e-6
#depth = 0.0127
#depthRange = (0, 0.02)
#offset = 256
#size = 512

wavelength = 450e-9
pixelSize = 1.5e-6
depthRange = (0, 0.002)
nDepths = 40
offset = 100
size = 190

holo = pyh.Holo(pyh.INLINE_MODE, wavelength, pixelSize)
holo.set_background(background)

midPoint = np.shape(hologram)[0] / 2
trialDepths = np.linspace(depthRange[0], depthRange[1], 20)



shiftx = np.zeros(len(trialDepths))
shifty = np.zeros(len(trialDepths))
maxVal = np.zeros(len(trialDepths))


for idx, depth in enumerate(trialDepths):
    
    holo.set_depth(depth)
    refocusImg = holo.process(hologram)
    fourierPlane = np.fft.fftshift(np.fft.fft2(refocusImg))
    
    leftRoi = Roi(midPoint - offset - size / 2, midPoint - size / 2,size,size )
    rightRoi = Roi(midPoint + offset - size / 2 + 1, midPoint - size / 2, size,size)
    
    leftBlock = leftRoi.clear_outside(fourierPlane)
    rightBlock = rightRoi.clear_outside(fourierPlane)
    
    
    #plt.imshow(np.abs(fourierPlane), cmap = 'gray', vmax = 100000)
    
    #plt.figure()
    #plt.imshow((np.abs(leftBlock)), cmap = 'gray')
    
    #plt.figure()
    #plt.imshow(np.abs(rightBlock), cmap = 'gray')
    
    leftFFT = (np.fft.fft2(leftBlock))
    rightFFT = (np.fft.fft2(rightBlock))
    
    reconFFT = leftFFT + rightFFT
    
    leftFFT = np.abs(leftFFT)
    rightFFT = np.abs(rightFFT)

    
    #leftFFT = cv.blur(leftFFT, (5,5))
    #rightFFT = cv.blur(rightFFT, (5,5))
    
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle(depth)
    axs[0,0].imshow(np.abs(leftFFT), cmap='gray')
    
   # plt.figure()
    axs[0,1].imshow(np.abs(rightFFT), cmap='gray')
    axs[1,0].imshow(np.abs(reconFFT), cmap='gray')
    axs[1,1].imshow(np.abs(refocusImg), cmap='gray')

    shifts = pyb.SuperRes.find_shift(leftFFT, rightFFT, size / 4, size / 2, 1)
    
    maxVal[idx] = np.sum(np.dot(leftFFT, rightFFT))
    shiftx[idx], shifty[idx] = shifts
    
plt.figure()
plt.plot(trialDepths, (shiftx))
plt.xlabel('Depth')
plt.ylabel('Shift')
plt.title("X Shift")


plt.figure()
plt.plot(trialDepths, (maxVal))
plt.xlabel('Depth')
plt.ylabel('Shift')
plt.title("Max")




# Test found focus
foundDepth = 0.00105 # trialDepths[np.argmax(shiftx)]
holo.set_depth(foundDepth)
focusImg = np.abs(holo.process(hologram))

plt.figure(dpi=150)
plt.imshow(focusImg, cmap='gray')
plt.title("Autofocused Image")




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