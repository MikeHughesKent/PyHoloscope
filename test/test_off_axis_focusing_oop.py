# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography with numerical refocusing
functionality of PyHoloscope using OOP (Holo)

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import context

from matplotlib import pyplot as plt

import time
import math

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6

# Load images
hologram = pyh.load_image("test data\\paramecium_oa_oof.tif")
background = pyh.load_image("test data\\paramecium_oa_oof_background.tif")

# Create object
holo = pyh.Holo(pyh.OFFAXIS_MODE, wavelength, pixelSize)

holo.set_background(background)
holo.auto_find_off_axis_mod()           # Finds modulation frequency
holo.off_axis_background_field()        # Processes background image to obtain background phase
holo.relativePhase = True
holo.stablePhase = False
holo.refocus = True
holo.depth = -0.0012

reconField = holo.process(hologram)

# These lines can be uncommented to dump a depth stack to test.tif
#refocusField = mHolo.refocus(hologram)
#depthStack = mHolo.depth_stack(reconField, (-0.0001,0.0001), 100)
#depthStack.writeIntensityToTif('test.tif')

# Display intensity and phase
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Intensity')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase')

DIC = pyh.synthetic_DIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = pyh.phase_gradient(reconField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')

# Unwrap phase
phaseUnwrapped = pyh.phase_unwrap(pyh.phase(reconField))

plt.figure(dpi = 150)
plt.imshow(phaseUnwrapped)
plt.title('Phase Unwrapped')

# Remove image phase tilt
tilt = pyh.obtain_tilt(phaseUnwrapped)
phaseUntilted = phaseUnwrapped - tilt

plt.figure(dpi = 150)
plt.imshow(phaseUntilted)
plt.title('Tilt Removed')

