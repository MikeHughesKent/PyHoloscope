# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography with numerical refocusing
functionality of PyHoloscope using direct calling of lower-level functions.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import context

from matplotlib import pyplot as plt

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
gridSize = 1024
depth = -0.0012

# Load images
hologram = pyh.load_image("test data\\paramecium_oa_oof.tif")
background = pyh.load_image("test data\\paramecium_oa_oof_background.tif")

# Determine Modulation
cropCentre = pyh.off_axis_find_mod(background)
cropRadius = pyh.off_axis_find_crop_radius(background)

# Remove modulation    
reconField = pyh.off_axis_demod(hologram, cropCentre, cropRadius)
backgroundField = pyh.off_axis_demod(background, cropCentre, cropRadius)

# Apply background correction 
correctedField = pyh.relative_phase(reconField, backgroundField)

prop = pyh.propagator(cropRadius * 2, wavelength, pixelSize / (cropRadius * 2) * gridSize, depth)
refocusedField = pyh.refocus(correctedField, prop)

# These lines can be uncommented to dump a depth stack to test.tif
#refocusField = mHolo.refocus(hologram)
#depthStack = mHolo.depth_stack(reconField, (-0.0001,0.0001), 100)
#depthStack.writeIntensityToTif('test.tif')

# Unwrap phase
phaseUnwrapped = pyh.phase_unwrap(pyh.phase(refocusedField))
tilt = pyh.obtain_tilt(phaseUnwrapped)

# Remove image phase tilt
phaseUntilted = phaseUnwrapped - tilt

# Display intensity and phase
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Intensity')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase')

plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(refocusedField), cmap = 'gray')
plt.title('Refocused Intensity')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(refocusedField), cmap = 'twilight')
plt.title('Refocused Phase (Wrapped)')

plt.figure(dpi = 150)
plt.imshow(phaseUnwrapped)
plt.title('Refocused Phase Unwrapped')

plt.figure(dpi = 150)
plt.imshow(phaseUntilted)
plt.title('Refocused Phase Unwrapped, Tilt Removed')

DIC = pyh.synthetic_DIC(refocusedField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = pyh.phase_gradient(refocusedField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')