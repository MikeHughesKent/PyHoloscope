# -*- coding: utf-8 -*-
"""
Example of processing off axis holography with numerical refocusing
functionality of pyholoscope using OOP

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

# Load images
hologram = pyh.load_image("..\\test\\test data\\paramecium_oa_oof.tif")
background = pyh.load_image("..\\test\\test data\\paramecium_oa_oof_background.tif")

# Create Holo object
holo = pyh.Holo(pyh.OFFAXIS_MODE, wavelength, pixelSize)

holo.set_background(background)         # Supply the background image/phase
holo.auto_find_off_axis_mod()           # Finds modulation frequency
holo.off_axis_background_field()        # Processes background image to obtain background phase
holo.relativePhase = True               # We will remove the background phase
holo.refocus = True                     # We will numerically refocus
holo.depth = -0.0012                    # Refocus distance in m


# In a single step we remove the off axis modulation and refocus
reconField = holo.process(hologram)


# Display intensity and phase
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Intensity')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase')


# Create a DIC style image from the phase
DIC = pyh.synthetic_DIC(reconField, shearAngle = 0)

plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')


# Create a phase gradient image
phaseGrad = pyh.phase_gradient(reconField)

plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')


# Unwrap phase
phaseUnwrapped = pyh.phase_unwrap(pyh.phase(reconField))

plt.figure(dpi = 150)
plt.imshow(phaseUnwrapped)
plt.title('Unwrapped Phase')


# Detect a global tilt in the phase and remove it (note we must supply the
#unwrapped phase here)
tilt = pyh.obtain_tilt(phaseUnwrapped)
phaseUntilted = pyh.relative_phase(phaseUnwrapped, tilt)

plt.figure(dpi = 150)
plt.imshow(phaseUntilted)
plt.title('Unwrapped Phase with Tilt Removed')

