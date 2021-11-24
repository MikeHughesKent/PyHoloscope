# -*- coding: utf-8 -*-
"""
Tests off-axis holography functionality.

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

import context    # relative paths

import PyHoloscope as holo
import PyHoloscope.sim as sim


# Experimental Parameters
wavelength = 630e-9
pixelSize = .3e-6


# Load images
hologram = cv.imread("test data\\tissue_paper_oa.tif",-1)
background = cv.imread("test data\\tissue_paper_oa_background.tif",-1)


# Determine Modulation
cropCentre = holo.offAxisFindMod(background)
cropRadius = holo.offAxisFindCropRadius(background)


print('Estimated tilt angle:', round(holo.offAxisPredictTiltAngle(background, wavelength, pixelSize) * 180 / math.pi,1), ' degrees')

    
reconField = holo.offAxisDemod(hologram, cropCentre, cropRadius)
backgroundField = holo.offAxisDemod(background, cropCentre, cropRadius)


correctedField = holo.relativePhase(reconField, backgroundField)
relativeField  = holo.relativePhaseROI(correctedField, holo.roi(20,20,5,5))


plt.figure(dpi = 150)
plt.imshow(hologram, cmap = 'gray')
plt.title('Hologram')

plt.figure(dpi = 150)
plt.imshow(np.angle(reconField), cmap = cmocean.cm.phase)
plt.title('Phase')

plt.figure(dpi = 150)
plt.imshow(np.angle(correctedField), cmap = cmocean.cm.phase)
plt.title('Corrected Phase')

plt.figure(dpi = 150)
plt.imshow(np.angle(relativeField), cmap = cmocean.cm.phase)
plt.title('Relative Phase')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconField), cmap = 'gray')
plt.title('Intensity')

DIC = holo.syntheticDIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = holo.phaseGradient(correctedField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')





