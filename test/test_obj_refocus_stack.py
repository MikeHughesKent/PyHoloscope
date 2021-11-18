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
imgSize = 400
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

depthRange = [0, 0.002]
nDepths = 100

t0 = time.time()
depthStack = hp.depthStack(holoProc, depthRange, nDepths)
print("Time to build depth stack (s): ", round(time.time() - t0,4))

t0 = time.time()
img = depthStack.getDepthIntensity(0.00053)
print("Time to fetch image (s): ", round(time.time() - t0,6))
plt.figure()
plt.imshow(img, cmap='gray')

t0 = time.time()
depthStack.writeIntensityToTif('test.tif')
print("Time to write TIF stack (s): ", round(time.time() - t0,6))

t0 = time.time()
depthStack.writePhaseToTif('test_phase.tif')
print("Time to write TIF stack for Phase (s): ", round(time.time() - t0,6))




