# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography with numerical refocusing
functionality of PyHoloscope using Holo class.

"""

import context

from matplotlib import pyplot as plt

from pathlib import Path

import time
import math

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = -0.0012

# Load images
hologram = pyh.load_image(Path('test data/paramecium_oa_oof.tif'))
background = pyh.load_image(Path('test data/paramecium_oa_oof_background.tif'))

# Create object
holo = pyh.Holo(mode = pyh.OFF_AXIS, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                background = background,
                autoWindow = False,
                relativePhase = True,
                refocus = True,
                depth = depth)

holo.calib_off_axis()        # Finds modulation frequency and background field

t1 = time.perf_counter()
reconField = holo.process(hologram)
print(f"Demodulation and refocusing time: {round((time.perf_counter() - t1) * 1000)} ms")

# Display intensity and phase
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray', interpolation='none')
plt.title('Intensity')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight', interpolation='none')
plt.title('Phase')

DIC = pyh.synthetic_DIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray', interpolation='none')
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

