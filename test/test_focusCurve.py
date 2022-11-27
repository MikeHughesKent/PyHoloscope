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
import pybundle


hologram = cv.imread(r"C:\Users\AOG\OneDrive - University of Kent\Experimental\Holography\Inline Bundle Holography\Slides with distance 26_09_22\data\Paramecium\2.tif")
#hologram = cv.imread("test data\\usaf_holo.tif")
hologram = hologram[:,:,1].astype('double')


background = cv.imread(r"C:\Users\AOG\OneDrive - University of Kent\Experimental\Holography\Inline Bundle Holography\Slides with distance 26_09_22\data\Paramecium\background.tif")
#background = cv.imread('test data\\usaf_back.tif')
background = background[:,:,1].astype('double')

# Bundle pattern removal
loc = (640,512,512)      # Fibre bundle location (x,y,radius)
imgSize = 512            # Image size after fibre core removal
rad = imgSize/2          # Image radius after fibre core removal

# Holography
skinThickness = 20       # Feathering of circular window
wavelength = 0.45e-6                  # Blue LED
pixelSize = 0.44e-6 / imgSize * 1024

# For Propagator LUT
depthRange = (0.0002, 0.001)
nDepths = 100
focusMetric = 'DarkFocus'

# Create masks and propagator
loc = pybundle.find_bundle(background)
mask = pybundle.get_mask(hologram, loc)
window = holo.circ_cosine_window(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = pybundle.crop_filter_mask(hologram, loc, mask, 3, resize = imgSize)
backgroundProc = pybundle.crop_filter_mask(background, loc, mask, 3, resize = imgSize)



t0 = time.time()
focusCurve, depths = holo.focus_score_curve(holoProc, wavelength, pixelSize, depthRange, nDepths, focusMetric, background=backgroundProc, window = window)
t1 = time.time()
print("Focus curve generation time:", round(t1-t0,3))

plt.figure()
plt.plot(depths * 1000, focusCurve)
plt.xlabel('Depth (mm)')
plt.ylabel('Focus Score')

depth = 0.00035
prop = holo.propagator(imgSize, wavelength, pixelSize, depth)
refocus = holo.refocus(holoProc - backgroundProc, prop)
plt.figure()
plt.imshow(np.abs(refocus), cmap='gray')