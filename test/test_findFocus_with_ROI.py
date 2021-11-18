# -*- coding: utf-8 -*-
"""
Tests finding focus of a ROI both by refocusing whole image and also
by refocusing just the ROI with a surrounding margin for speed.

@author: Mike Hughes
"""

import random
import cv2 as cv
import context


import PyHoloscope as holo

from pybundle import PyBundle
import time

hologram = cv.imread("test data\\microspheres_holo.tif")
#hologram = cv.imread("test data\\usaf_holo.tif")

hologram = hologram[:,:,1]
background = cv.imread('test data\\microspheres_back.tif')
#background = cv.imread('test data\\usaf_back.tif')

background = background[:,:,1]


loc = (640,512,512)
imgSize = 300
rad = imgSize/2
skinThickness = 20
wavelength = 0.45e-6
pixelSize = 0.44e-6 / imgSize * 1024
depthRange = (0.0004,0.0007)

roi = holo.roi(100,100,50,50)

# Create masks and propagator
mask = PyBundle.getMask(hologram, loc)
window = holo.circCosineWindow(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)

t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Sobel', background=backgroundProc, window = window, roi = roi, margin = 20)
t1 = time.time()
print("Time to find focus using ROI and margin:", round(t1-t0,3))
print("Found depth:", round(depth,5))

t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Sobel', background=backgroundProc, window = window, roi = roi)
t1 = time.time()
print("Time to find focus using FFT whole image:", round(t1-t0,3))
print("Found depth:", round(depth,5))
