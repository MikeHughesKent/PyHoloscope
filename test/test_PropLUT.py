# -*- coding: utf-8 -*-
"""
Tests finding focus using pre-calculated look up table (LUT) of propagators
for depths. This approxiamtely halved the time to locate the optical focal
depth.

@author: Mike Hughes
"""

import random
import cv2 as cv
import context


import pyHoloscope as holo

from pybundle import PyBundle
import time

hologram = cv.imread("test data\\microspheres_holo.tif")
#hologram = cv.imread("test data\\usaf_holo.tif")

hologram = hologram[:,:,1]
background = cv.imread('test data\\microspheres_back.tif')
#background = cv.imread('test data\\usaf_back.tif')

background = background[:,:,1]


loc = (640,512,512)
imgSize = 200
rad = imgSize/2
skinThickness = 20
wavelength = 0.45e-6
pixelSize = 0.44e-6 / imgSize * 1024
depth = 0.00075

# Create masks and propagator
mask = PyBundle.getMask(hologram, loc)
window = holo.circCosineWindow(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)


# For Propagator LUT
depthRange = (0.0002, 0.001)
nDepths = 100

t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Sobel', background=backgroundProc, window = window)
t1 = time.time()
print("Find focus using propagator calculation time:", round(t1-t0,3))
print("Found depth:", round(depth,5))

t0 = time.time()
propLUT = holo.PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths)
t1 = time.time()
print("Propagator LUT Build time:", round(t1-t0,3))


t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Sobel', background=backgroundProc, window = window, propagatorLUT = propLUT)
t1 = time.time()
print("Find focus using propagator LUT time:", round(t1-t0,3))
print("Found depth:", round(depth,5))

t0 = time.time()
for i in range(10000):
    depth = random.random() * 0.0005 + 0.0003
    prop = propLUT.propagator(depth)
t1 = time.time()
print("Time to look up propagator in table: ", round( (t1-t0) / 10000,7))