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
pixelSize = .3e-6

# Load images
hologram = pyh.load_image(Path('test data/tissue_paper_oa.tif'))
background = pyh.load_image(Path('test data/tissue_paper_oa_background.tif'))

# Create object
holo = pyh.Holo(pyh.OFF_AXIS, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                background = background,
                relativePhase = False)
                    

# Find modulation frequency 
holo.calib_off_axis(background)     

# Remove modulation
reconField = holo.process(hologram)
    
reconFieldCorrected = pyh.relative_phase_self(reconField, roi = pyh.Roi(40,40,10,10))

print(pyh.mean_phase(reconFieldCorrected))

plt.figure(dpi = 150)
plt.imshow(np.angle(reconField), cmap = 'twilight')
plt.title('Phase uncorrected')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconField), cmap = 'gray')
plt.title('Amplitude uncorrected')

plt.figure(dpi = 150)
plt.imshow(np.angle(reconFieldCorrected), cmap='twilight')
plt.title('Phase corrected')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconFieldCorrected), cmap = 'gray')
plt.title('Amplitude corrected')