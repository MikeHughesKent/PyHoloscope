# -*- coding: utf-8 -*-
"""
Tests sim modulation of PyHoloscope - simulation and recovery of phase in off-axis holography.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import math

import context    # Load paths

import pyholoscope as holo
import pyholoscope.sim as sim


# Simulated experimental parameters
wavelength = 450e-9
pixelSize = 0.44e-6
tiltAngle = 2    # reference beam tilt


# Create simulated background and object fields
backgroundField = np.ones((1024,1024), dtype = complex)

objectField = np.ones((1024,1024), dtype = complex)
objectField[0:512,:] = 2
phaseField = np.zeros((1024,1024))
phaseField[0:512,0:512] = 3.14150

objectField = objectField * np.exp(1j * phaseField)


# Simulate holograms from background and object
backgroundImage = sim.off_axis(backgroundField, wavelength, pixelSize, tiltAngle)
cameraImage = sim.off_axis(objectField, wavelength, pixelSize, tiltAngle)


# Using the peak in the FFT, determine the modulation frequency and hence the
# ROI to use
cropCentre = holo.off_axis_find_mod(backgroundImage)
cropRadius = holo.off_axis_find_crop_radius(backgroundImage)


# Off-axis reconstruction for background and object
backgroundField = holo.off_axis_demod(backgroundImage, cropCentre, cropRadius)
reconField = holo.off_axis_demod(cameraImage, cropCentre, cropRadius)


# Subtract background phase
correctedField = holo.relative_phase(reconField, backgroundField)


plt.figure(dpi = 300)
plt.title('Object Phase')
plt.imshow(holo.phase(objectField), cmap = 'twilight', vmin = 0, vmax = 2 * math.pi)

plt.figure(dpi = 300)
plt.title('Hologram')
plt.imshow(cameraImage, cmap = 'gray')

plt.figure(dpi = 300)
plt.title('Reconstructed Phase')
plt.imshow(holo.phase(correctedField), cmap = 'twilight', vmin = 0, vmax = 2 * math.pi)

plt.figure(dpi = 300)
plt.title('Reconstructed Amplitude')
plt.imshow(holo.amplitude(correctedField), cmap = 'gray')
