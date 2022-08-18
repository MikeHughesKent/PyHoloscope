# -*- coding: utf-8 -*-
"""
Tests finding focus using a coarse search to first narrow down the search
range. Compares with an exhaustive check of all depths and a search without the
prior coarse search (which is more likely to get caught in local minima).

For speed, a propagator LUT is used, and refocusing is only performed using
a margin around the ROI.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import time
import numpy as np

import cv2 as cv

import context               # Loads paths

import PyHoloscope as holo
from pybundle import PyBundle


hologramFile = 'test data\\microspheres_holo.tif'
backgroundFile = 'test data\\microspheres_back.tif'


hologram = cv.imread(hologramFile)[:,:,0]
background = cv.imread(backgroundFile)[:,:,0]


# Bundle pattern removal
loc = (640,512,512)      # Fibre bundle location (x,y,radius)
imgSize = 300            # Image size after fibre core removal
rad = imgSize/2          # Image radius after fibre core removal

# Holography
skinThickness = 20       # Feathering of circular window
wavelength = 0.45e-6                  # Blue LED
pixelSize = 0.44e-6 / imgSize * 1024

# Focus finding
roi = holo.roi(100,100,50,50) # Region of interest containing object to be focused on
depthRange = (0.0004,0.002)   # Refocusing only within this depth range
marginSize = 20               # When ROI is used, only an area with this margin 
                              #    around the ROI will be refocused
nLUTDepths =  200                # Number of depths in propagator LUT
focusMetric = 'DarkFocus'

coarseSearchInterval = 10     # Coarse search will be performed at this number of
                              # points equally spaced over depth range

# Create masks and propagator
mask = PyBundle.getMask(hologram, loc)
window = holo.circCosineWindow(imgSize,rad, skinThickness)


# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 3, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 3, resize = imgSize)


# Build propagator table for subsequent focusing speed
propLUTRoi = holo.PropLUT(roi.width + marginSize *2 , wavelength, pixelSize, depthRange, nLUTDepths)


# Slow exhaustive search for focus
t0 = time.time()
focusCurve, depths = holo.focusScoreCurve(holoProc, wavelength, pixelSize, depthRange, nLUTDepths, focusMetric, roi = roi, background=backgroundProc, window = window)
depth = depths[np.argmin(focusCurve)]
t1 = time.time()
print("Time for exhaustive search:", round(1000 * (t1-t0),1), 'ms')
print("Found depth using exhaustive search:", round(1000 * depth,2), ' mm')
print("\n")

# Fast refocusing ROI only, propagator LUT
t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, focusMetric, background=backgroundProc, window = window, roi = roi, margin = marginSize, propagatorLUT = propLUTRoi)
t1 = time.time()
print("Time to find focus using only fine search:", round(1000 * (t1-t0),1), 'ms')
print("Fast found depth:", round(1000 * depth,2), ' mm')
print("\n")

# Refocusing ROI only, propagator LUT, coarse-first search
t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, focusMetric, background=backgroundProc, coarseSearchInterval = coarseSearchInterval, window = window, roi = roi, margin = marginSize, propagatorLUT = propLUTRoi)
t1 = time.time()
print("Time to find focus using coarse Search first:", round(1000 * (t1-t0),1), 'ms')
print("Fast found depth with coarse search:", round(1000 * depth,2), ' mm')