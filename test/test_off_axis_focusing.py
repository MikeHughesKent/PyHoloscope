# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography with numerical refocusing
functionality of PyHoloscope

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


import sys

import context

import PyHoloscope as holo

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6


# Load images
hologram = cv.imread("test data\\paramecium_oa_oof.tif",-1)
background = cv.imread("test data\\paramecium_oa_oof_background.tif",-1)
hologram = hologram[0:1024,0:1024]
background = background[0:1024,0:1024]

# Create object
mHolo = holo.Holo(holo.OFFAXIS_MODE, wavelength, pixelSize)


mHolo.set_background(background)
mHolo.auto_find_off_axis_mod()           # Finds modulation frequency
mHolo.off_axis_background_field()        # Processes background image to obtain background phase
mHolo.relativePhase = True
mHolo.stablePhase = False
mHolo.refocus = True
mHolo.depth = -0.000086

reconField = mHolo.process(hologram)

# These lines can be uncommented to dump a depth stack to test.tif
#refocusField = mHolo.refocus(hologram)
#depthStack = mHolo.depth_stack(reconField, (-0.0001,0.0001), 100)
#depthStack.writeIntensityToTif('test.tif')

# Display intensity and phase
plt.figure(dpi = 150)
plt.imshow(np.angle(reconField), cmap = cmocean.cm.phase)
plt.title('Phase')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconField), cmap = 'gray')
plt.title('Intensity')

DIC = holo.synthetic_DIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = holo.phase_gradient(reconField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')


# Unwrap phase
phaseUnwrapped = holo.phase_unwrap(np.angle(reconField))

plt.figure(dpi = 150)
plt.imshow(phaseUnwrapped)
plt.title('Phase Unwrapped')



# Remove image phase tilt
tilt = holo.obtain_tilt(phaseUnwrapped)
phaseUntilted = phaseUnwrapped - tilt

plt.figure(dpi = 150)
plt.imshow(phaseUntilted)
plt.title('Tilt Removed')

