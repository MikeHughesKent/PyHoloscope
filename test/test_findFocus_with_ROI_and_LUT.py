# -*- coding: utf-8 -*-
"""
Tests finding focus using focus metric. Measures time taken when refocusing
whole image, refocusing a region around the ROI only, and with a propagator LUT.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import time
import context    # Loads paths

import cv2 as cv

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
depthRange = (0.0002,0.0008)  # Refocusing only within this depth range
marginSize = 20               # When ROI is used, only an area with this margin 
                                   # around the ROI will be refocused
nDepths =  200               # Number of depths in propagator LUT
focusMetric = 'DarkFocus'

# Create masks and propagator
mask = PyBundle.getMask(hologram, loc)
window = holo.circCosineWindow(imgSize,rad, skinThickness)


# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)


# Build propagator table for subsequent focusing speed
propLUT = holo.PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths)
propLUTRoi = holo.PropLUT(roi.width + marginSize *2 , wavelength, pixelSize, depthRange, nDepths)


# Refocusing whole image, no LUT
t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, focusMetric, background=backgroundProc, window = window, roi = roi)
t1 = time.time()
print("Time to find focus using whole image, no LUT:", round(1000 * (t1-t0),1), 'ms')
print("Found depth:", round(1000 * depth,2), ' mm')


# Refocusing whole image, using LUT
t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, focusMetric, background=backgroundProc, window = window, roi = roi, margin = marginSize)
t1 = time.time()
print("Time to find focus using ROI and margin, no LUT:", round(1000 * (t1-t0),1), 'ms')
print("Found depth:", round(1000 * depth,2), ' mm')

# Refocusing ROI only, no LUT
t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, focusMetric, background=backgroundProc, window = window, roi = roi, propagatorLUT = propLUT)
t1 = time.time()
print("Time to find focus using whole image and propagator LUT:", round(1000 * (t1-t0),1), 'ms')
print("Found depth:", round(1000 * depth,2), ' mm')

# Refocusing ROI only, LUT
t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, focusMetric, background=backgroundProc, window = window, roi = roi, margin = marginSize, propagatorLUT = propLUTRoi)
t1 = time.time()
print("Time to find focus using ROI and propagator LUT:", 1000 * round(t1-t0,3), 'ms')
print("Found depth:", round(1000 * depth,2), ' mm')
