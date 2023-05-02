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

sys.path.append("..\src")
import pyholoscope as pyh

hologram = pyh.load_image(r"..\\test\\test data\\inline_example_holo.tif")
background = pyh.load_image(r"..\\test\\test data\\inline_example_back.tif")


# Holography
skinThickness = 20       # Feathering of circular window

wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0127

depthRange = (0.001, 0.04)
nDepths = 40
depths = np.linspace(depthRange[0], depthRange[1], nDepths)

focusMetrics = ['Brenner', 'Peak', 'Sobel', 'SobelVariance', 'Var', 'DarkFocus']

roi = pyh.Roi(390,280,250,250)

holo = pyh.Holo(pyh.INLINE_MODE, wavelength, pixelSize)
holo.set_background(background)
holo.set_depth(depth)
refocusImg = holo.process(hologram)

refocsStack = holo.depth_stack(hologram, depthRange, nDepths)
refocsStack.write_intensity_to_tif('stack.tif')

plt.figure()
plt.imshow(pyh.amplitude(refocusImg), cmap='gray')


plt.figure()
plt.xlabel('Depth (mm)')
plt.ylabel('Focus Score')


focusCurve = np.zeros((len(focusMetrics),len(depths)))
for metricIdx, focusMetric in enumerate(focusMetrics):
    for idx, depth in enumerate(depths):   
    
    
        cImg = pyh.amplitude(refocsStack.get_index(idx))
        cImg = roi.crop(cImg)
        focusCurve[metricIdx, idx] = pyh.focus_score(cImg, focusMetric)
    
    
    focusCurveNorm = focusCurve[metricIdx,:] - np.min(focusCurve[metricIdx,:])
    focusCurveNorm = focusCurveNorm / np.max(focusCurveNorm)
    plt.plot(depths * 1000, focusCurveNorm, label = focusMetric)

plt.legend()


# for idx, depth in enumerate(depths):   


#     cImg = pyh.amplitude(refocsStack.get_index(idx))
#     cImg = roi.crop(cImg)
#     focusCurve[idx] = pyh.focus_score(cImg, 'DarkFocus')


#focusCurve = focusCurve - np.min(focusCurve)
#focusCurve = focusCurve / np.max(focusCurve)
#plt.plot(depths * 1000, focusCurve, label = focusMetric)


holo.set_use_numba(True)
t1 = time.perf_counter()
depth = holo.auto_focus(hologram, depthRange = depthRange, method = 'DarkFocus', roi = roi, margin = None)
print("Refocus time", time.perf_counter() - t1)

holo.set_depth(depth)
refocusImg = holo.process(hologram)
plt.imshow(pyh.amplitude(refocusImg), cmap='gray')

# depth = 0.00035
# prop = pyh.propagator(imgSize, wavelength, pixelSize, depth)
# refocus = pyh.refocus(holoProc - backgroundProc, prop)
# plt.figure()
# plt.imshow(pyh.amplitude(refocus), cmap='gray')