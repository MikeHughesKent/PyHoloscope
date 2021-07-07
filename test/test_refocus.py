# -*- coding: utf-8 -*-
"""
Tests numerical refocus of inline hologram.

@author: Mike Hughes
"""

from matplotlib import pyplot as plt
import numpy as np
import time

import cv2 as cv

import context
import PyHoloscope as holo
from pybundle import PyBundle


hologram = cv.imread("test data\\microspheres_holo.tif")
hologram = hologram[:,:,1]
background = cv.imread('test data\\microspheres_back.tif')
background = background[:,:,1]


loc = (640,512,512)
imgSize = 300
rad = imgSize/2
skinThickness = 20
wavelength = 0.45e-6
pixelSize = 0.44e-6 / imgSize * 1024
depth = 0.00058

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
print("Refocus Time:", t1-t0)
plt.imshow(refocImg, cmap='gray')
