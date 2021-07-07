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
from PyHoloscope import roi
from pybundle import PyBundle
import time

#hologram = cv.imread("test data\\usaf_holo.tif")
hologram = cv.imread("test data\\microspheres_holo.tif")

hologram = hologram[:,:,1]
#background = cv.imread('test data\\usaf_back.tif')
background = cv.imread('test data\\microspheres_back.tif')

background = background[:,:,1]

loc = (640,512,512)
imgSize = 400
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



ROI = roi(50,50,200,200)

depthRange = (0.0002, 0.001)
nPoints = 100
#print(holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Brenner', roi = ROI, margin = 50, background=backgroundProc))

t0 = time.time()
depth = holo.findFocus(holoProc, wavelength, pixelSize, depthRange, 'Peak', background=backgroundProc, window = window)
t1 = time.time()
print("Find focus time:", t1-t0)
print(depth)
#focusCurve, depths = holo.focusScoreCurve(holoProc, wavelength, pixelSize, depthRange, nPoints, 'Brenner', window = window, background=backgroundProc)


#val, idx = min((val, idx) for (idx, val) in enumerate(focusCurve))

#depth = depths[idx]
#plt.imshow(holoProc, cmap='gray')
#plt.imshow(np.angle(prop), cmap='gray')
#plt.imshow(refocImg, cmap='gray')
#plt.plot(depths * 1000, focusCurve)

prop = holo.propagator(imgSize, pixelSize, wavelength, depth)



t0 = time.time()
refocImg = np.abs(holo.refocus(holoProc, prop, background = backgroundProc, window = window))
t1 = time.time()
print("Refocus Time:", t1-t0)
print(holo.focusScore(refocImg, 'Brenner'))
plt.imshow(refocImg, cmap='gray')
