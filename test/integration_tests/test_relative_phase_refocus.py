# -*- coding: utf-8 -*-
"""
Tests relative phase part of off axis holography functionality of PyHoloscope

"""

from matplotlib import pyplot as plt

import numpy as np
import time
from pathlib import Path

import context              # Load paths       

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = -0.0012

# Load images
hologram = pyh.load_image(Path('test data/paramecium_oa_oof.tif'))
background = pyh.load_image(Path('test data/paramecium_oa_oof_background.tif'))


# Create object
holo = pyh.Holo(pyh.OFF_AXIS, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                background = background,
                relativePhase = True,
                refocus = True,
                depth = depth)
                    

# Find modulation frequency 
holo.calib_off_axis(background)     

# Remove modulation
reconField = holo.process(hologram)
    
# Make phase relative to a region of the image
reconFieldCorrected = pyh.relative_phase_self(reconField, roi = pyh.Roi(40,40,10,10))


plt.figure(dpi = 150)
plt.imshow(np.angle(reconField), cmap = 'twilight', interpolation='none')
plt.title('Phase uncorrected')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconField), cmap = 'gray', interpolation='none')
plt.title('Amplitude uncorrected')

plt.figure(dpi = 150)
plt.imshow(np.angle(reconFieldCorrected), cmap='twilight', interpolation='none')
plt.title('Phase corrected')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconFieldCorrected), cmap = 'gray', interpolation='none')
plt.title('Amplitude corrected')