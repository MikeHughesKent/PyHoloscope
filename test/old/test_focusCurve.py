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

import context                  # Loads paths

import PyHoloscope as holo
from pybundle import PyBundle


hologram = cv.imread("test data\\microspheres_holo.tif")
#hologram = cv.imread("test data\\usaf_holo.tif")
hologram = hologram[:,:,1]


background = cv.imread('test data\\microspheres_back.tif')
#background = cv.imread('test data\\usaf_back.tif')
background = background[:,:,1]

# Bundle pattern removal
loc = (640,512,512)      # Fibre bundle location (x,y,radius)
imgSize = 300            # Image size after fibre core removal
rad = imgSize/2          # Image radius after fibre core removal

# Holography
skinThickness = 20       # Feathering of circular window
wavelength = 0.45e-6                  # Blue LED
pixelSize = 0.44e-6 / imgSize * 1024

# For Propagator LUT
depthRange = (0.0002, 0.001)
nDepths = 100
focusMetric = 'DarkFocus'
depth = 0.00075

# Create masks and propagator
mask = PyBundle.getMask(hologram, loc)
window = holo.circCosineWindow(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 3, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 3, resize = imgSize)



t0 = time.time()
focusCurve, depths = holo.focusScoreCurve(holoProc, wavelength, pixelSize, depthRange, nDepths, focusMetric, background=backgroundProc, window = window)
t1 = time.time()
print("Focus curve generation time:", round(t1-t0,3))
print("Found depth:", round(depth,5))

plt.figure()
plt.plot(depths * 1000, focusCurve)
plt.xlabel('Depth (mm)')
plt.ylabel('Focus Score')

