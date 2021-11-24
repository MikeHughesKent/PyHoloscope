# -*- coding: utf-8 -*-
"""
Tests using object oriented functionality of PyHoloscope to generate a 
stack of images at different focus depths.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time

import cv2 as cv

import context     # Load paths

import PyHoloscope as holo
from pybundle import PyBundle


hologramFile = 'test data\\microspheres_holo.tif'
backgroundFile = 'test data\\microspheres_back.tif'

outputAmplitudeFile = 'output\\test_stack_amp.tif'
outputPhaseFile = 'output\\test_stack_phase.tif'


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


# Pre-process bundle images
mask = PyBundle.getMask(hologram, loc)
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)


# Create object and set some parameters
hp = holo.Holo(holo.INLINE_MODE, wavelength, pixelSize)
hp.setBackground(backgroundProc)
hp.setWindow(backgroundProc, rad, skinThickness)


# Specify stack of refocused images
depthRange = [0, 0.002]
nDepths = 100

t0 = time.time()
depthStack = hp.depthStack(holoProc, depthRange, nDepths)
print("Time to build depth stack (s): ", round(time.time() - t0,4))

t0 = time.time()
img = depthStack.getDepthIntensity(0.00053)
print("Time to fetch image from stack (s): ", round(time.time() - t0,6))
plt.figure()
plt.imshow(img, cmap='gray')

t0 = time.time()
depthStack.writeIntensityToTif(outputAmplitudeFile)
print("Time to write TIF stack for amplitude (s): ", round(time.time() - t0,6))

t0 = time.time()
depthStack.writePhaseToTif(outputPhaseFile)
print("Time to write TIF stack for phase (s): ", round(time.time() - t0,6))




