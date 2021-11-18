# -*- coding: utf-8 -*-
"""
Tests sim functionality

@author: Mike Hughes
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import math

import cmocean
import cv2 as cv

import context
from pybundle import PyBundle

import PyHoloscope as holo
import PyHoloscope.sim as sim


wavelength = 630e-9
pixelSize = .3e-6
tiltAngle = .2
cropRadius = 120

mHolo = holo.Holo(holo.INLINE_MODE, wavelength, pixelSize)


hologram = cv.imread("test data\\embryo_holo.png")
hologram = hologram[:,:,1]


background = cv.imread("test data\\embryo_back.png")
background = background[:,:,1]

mHolo.setBackground(background)
mHolo.autoFindOffAxisMod()
mHolo.cropRadius =120
mHolo.offAxisBackgroundField()
reconField = mHolo.offAxisRecon(hologram)





plt.figure(dpi = 150)
plt.imshow(hologram, cmap = 'gray')
plt.title('Hologram')

plt.figure(dpi = 150)
plt.imshow(np.angle(reconField), cmap = cmocean.cm.phase)
plt.title('Phase')



plt.figure(dpi = 150)
plt.imshow(np.abs(reconField), cmap = 'gray')
plt.title('Intensity')

DIC = holo.syntheticDIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = holo.phaseGradient(reconField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')


mHolo.setBackground(background)
mHolo.autoFindOffAxisMod()