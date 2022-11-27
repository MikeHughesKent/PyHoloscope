# -*- coding: utf-8 -*-
"""
Plot of focus score against refocus depth for different focus metrics.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import random
import time

import cv2 as cv

import context               # Loads paths

import PyHoloscope as holo


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

# For Propagator LUT
depthRange = (0.0002, 0.002)
nDepths = 100
focusMetric = 'DarkFocus'
depth = 0.00075


roi = holo.roi(100,100,50,50)    # Region of interest containing object to be focused on


# Create masks and propagator
mask = PyBundle.get_mask(hologram, loc)
window = holo.circ_cosine_window(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)

metrics = ['Peak', 'Sobel', 'Brenner', 'SobelVariance', 'DarkFocus']
nDepths =  200              # Number of depths in plot
propLUT = holo.PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths)


for metric in metrics:
    focusCurve, depths = holo.focusScoreCurve(holoProc, wavelength, pixelSize, depthRange, nDepths, metric, background=backgroundProc, window = window, propagatorLUT = propLUT)
    focusCurve = focusCurve - min(focusCurve)
    focusCurve = focusCurve / max(focusCurve)
    plt.plot(depths * 1000, focusCurve, label= metric)
    
plt.xlabel('Depth (mm)')
plt.ylabel('Normalised Focus Score')
plt.grid()
plt.legend()
plt.show()
