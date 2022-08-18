# -*- coding: utf-8 -*-
"""
Tests numerical refocus of inline hologram.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time

import cv2 as cv

import context      # Load paths
import PyHoloscope as holo
from pybundle import PyBundle


hologram = cv.imread("test data\\microspheres_holo.tif")[:,:,0]
background = cv.imread('test data\\microspheres_back.tif')[:,:,0]


# Bundle pattern removal
loc = (640,512,512)      # Fibre bundle location (x,y,radius)
imgSize = 300            # Image size after fibre core removal
rad = imgSize/2          # Image radius after fibre core removal

# Holography
skinThickness = 20       # Feathering of circular window
wavelength = 0.45e-6                  # Blue LED
pixelSize = 0.44e-6 / imgSize * 1024


depth = 0.00054

# Create mask and window
mask = PyBundle.getMask(hologram, loc)
window = holo.circCosineWindow(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)

# Propagator
prop = holo.propagator(imgSize, wavelength, pixelSize, depth)

t0 = time.time()
refocImg = np.abs(holo.refocus(holoProc, prop, background = backgroundProc, window = window))
t1 = time.time()
print("Refocus Time:", round(1000 * (t1-t0),1), 'ms')
plt.imshow(refocImg, cmap='gray')
