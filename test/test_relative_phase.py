# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography functionality of PyHoloscope

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt

import numpy as np
import time

import context              # Load paths       

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = .3e-6

# Load images
hologram = pyh.load_image("test data\\tissue_paper_oa.tif")
background = pyh.load_image("test data\\tissue_paper_oa_background.tif")

# Create object
holo = pyh.Holo(pyh.OFF_AXIS, 
                wavelength = wavelength, 
                pixelSize = pixelSize)
                    

# Find modulation frequency 
holo.calib_off_axis(background)     

# Remove modulation
reconField = holo.process(hologram)
backField = holo.process(background)


t1 = time.perf_counter()
reconFieldCorrected = pyh.relative_phase(reconField, backField)
print(f"Relative phase corretime time: {round((time.perf_counter() - t1) * 1000)} ms.")

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