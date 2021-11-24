# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography functionality

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import math

import cmocean
import cv2 as cv

import context              # Load paths       

import PyHoloscope as holo

# Experimental Parameters
wavelength = 630e-9
pixelSize = .3e-6


# Load images
hologram = cv.imread("test data\\tissue_paper_oa.tif",-1)
background = cv.imread("test data\\tissue_paper_oa_background.tif",-1)


# Create object
mHolo = holo.Holo(holo.INLINE_MODE, wavelength, pixelSize)


mHolo.setBackground(background)
mHolo.autoFindOffAxisMod()            # Finds modulation frequency
mHolo.offAxisBackgroundField()        # Processes background image to obtain background phase


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

