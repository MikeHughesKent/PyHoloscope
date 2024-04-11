# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography functionality of PyHoloscope

@author: Mike Hughes
Applied Optics Group
University of Kent
"""
import time

from matplotlib import pyplot as plt

import numpy as np

import context              # Load paths       

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = .3e-6

# Load images
hologram = pyh.load_image("test data\\tissue_paper_oa.tif")
background = pyh.load_image("test data\\tissue_paper_oa_background.tif") 

# Create the Holo object that will be used for demodulation
holo = pyh.Holo(pyh.OFF_AXIS, 
                wavelength = wavelength, 
                pixelSize = pixelSize, 
                background = background,
                cropMask = pyh.Holo.CIRCLE_COSINE,
                cropWindowSkinThickness = 10,
                relativePhase = True)               
                    
# Find modulation frequency and generate background and normalisation fields
# and create window
holo.calib_off_axis()   

# Remove modulation
t1 = time.perf_counter()
reconField = holo.process(hologram)
print(f"Off-axis demodulation time: {round((time.perf_counter() - t1) * 1000)} ms")

# Display results

plt.figure(dpi = 150)
plt.imshow(hologram, cmap = 'gray')
plt.title('Hologram')

plt.figure(dpi = 150)
plt.imshow(np.abs(reconField), cmap = 'gray')
plt.title('Amplitude')

plt.figure(dpi = 150)
plt.imshow(np.angle(reconField), cmap = 'twilight')
plt.title('Phase')

DIC = pyh.synthetic_DIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = pyh.phase_gradient(reconField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')

