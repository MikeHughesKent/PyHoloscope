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
windowRad = imgSize/2
windowSkinThickness = 20
wavelength = 0.45e-6
pixelSize = 0.44e-6 / imgSize * 1024
depth = 0.00058
mask = PyBundle.getMask(hologram, loc)

a = holo.roi(10,10,20,20)


# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)


hp = holo.Holo(wavelength, pixelSize)
hp.setBackground(backgroundProc)
hp.setWindow(backgroundProc, windowRad, windowSkinThickness)
hp.setDepth(depth)
hp.updatePropagator(holoProc)

t0 = time.time()
refocusComplex = hp.refocus(holoProc)
refocusImage = np.abs(refocusComplex)
print("Refocus time (s): ", round(time.time() - t0, 4))

hp.setFindFocusParameters('Sobel', None, None, [0.0002,0.001])

# Refocus
t0 = time.time()
depth = hp.findFocus(holoProc)
print("Find Focus Time (s): ", round(time.time() - t0, 4))

# Find focus
depthRange = [0.0002,0.001]
nDepths = 400
t0 = time.time()
hp.makePropagatorLUT(holoProc, depthRange, nDepths)
print("Build LUT Time (s): ", round(time.time() - t0, 4))

# Find focus using Propagator LUT
t0 = time.time()
depthUsingLUT = hp.findFocus(holoProc)
print("Find Focus Time  with LUT (s): ", round(time.time() - t0, 4))

plt.figure()
plt.imshow(refocusImage, cmap = 'gray')

