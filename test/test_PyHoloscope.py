# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:00:43 2021

@author: AOG
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import context


import PyHoloscope as holo
from pybundle import PyBundle
import time

hologram = cv.imread("test data\\usaf_holo.tif")
hologram = hologram[:,:,1]
background = cv.imread('test data\\usaf_back.tif')
background = background[:,:,1]

loc = (640,512,512)
imgSize = 300
rad = imgSize/2
skinThickness = 20
wavelength = 0.45e-6
pixelSize = 0.44e-6 / imgSize * 1024
depth = 0.00075

# Create masks and propagator
mask = PyBundle.getMask(hologram, loc)
prop = holo.propagator(imgSize, pixelSize, wavelength, depth)
window = holo.circCosineWindow(imgSize,rad, skinThickness)

# Pre-process bundle images
holoProc = PyBundle.cropFilterMask(hologram, loc, mask, 2.5, resize = imgSize)
backgroundProc = PyBundle.cropFilterMask(background, loc, mask, 2.5, resize = imgSize)

t0 = time.time()
refocImg = np.abs(holo.refocus(holoProc, prop, background = backgroundProc, window = window))
t1 = time.time()
print("Refocus Time:", t1-t0)
print(holo.focusScore(refocImg, 'Brenner'))
depthRange = (0.0002, 0.001)
print(holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Sobel', background=backgroundProc))

#plt.imshow(holoProc, cmap='gray')
#plt.imshow(np.angle(prop), cmap='gray')
plt.imshow(refocImg, cmap='gray')