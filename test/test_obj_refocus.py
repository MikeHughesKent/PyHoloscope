# -*- coding: utf-8 -*-
"""
Tests using the object oriented functionality of PyHoloscope to determine
the focal depth and refocus.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time

import cv2 as cv

import context        # Load paths

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


# Create masks for bundle
mask = PyBundle.getMask(hologram, loc)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)


# Create PyHoloscope object
hp = holo.Holo(holo.INLINE_MODE, wavelength, pixelSize)
hp.setBackground(backgroundProc)
hp.setWindow(backgroundProc, rad, skinThickness)
hp.updatePropagator(holoProc)


hp.setFindFocusParameters('DarkFocus', None, None, [0.0002,0.001])

# Find Focus
depthRange = [0.0002,0.001]
t0 = time.time()
depth = hp.findFocus(holoProc)
print("Find Focus Time (s): ", round(time.time() - t0, 4))

# Build propagator LUT
nDepths = 400
t0 = time.time()
hp.makePropagatorLUT(holoProc, depthRange, nDepths)
print("Build LUT Time (s): ", round(time.time() - t0, 4))

# Find focus using Propagator LUT
t0 = time.time()
depthUsingLUT = hp.findFocus(holoProc)
print("Find Focus Time  with LUT (s): ", round(time.time() - t0, 4))


# Refocus
hp.setDepth(depth)
t0 = time.time()
refocusComplex = hp.refocus(holoProc)
refocusImage = np.abs(refocusComplex)
print("Refocus time (s): ", round(time.time() - t0, 4))

plt.figure()
plt.imshow(refocusImage, cmap = 'gray')

