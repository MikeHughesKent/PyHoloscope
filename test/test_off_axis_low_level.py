# -*- coding: utf-8 -*-
"""
Tests off-axis holography functionality of PyHoloscope using low level functions.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt

import time
import math

import context                    # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = .6e-6

# Load images
hologram = pyh.load_image("test data\\tissue_paper_oa.tif")
background = pyh.load_image("test data\\tissue_paper_oa_background.tif")

# Determine Modulation
cropCentre = pyh.off_axis_find_mod(background)
cropRadius = pyh.off_axis_find_crop_radius(background)

# Remove modulation    
reconField = pyh.off_axis_demod(hologram, cropCentre, cropRadius)
backgroundField = pyh.off_axis_demod(background, cropCentre, cropRadius)

# Apply background correction and phase offset correction
correctedField = pyh.relative_phase(reconField, backgroundField)
relativeField = pyh.relative_phase_ROI(correctedField, pyh.Roi(20,20,45,45))

# Display results
plt.figure(dpi = 150)
plt.imshow(hologram, cmap = 'gray')
plt.title('Hologram')
    
plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(correctedField), cmap = 'twilight')
plt.title('Corrected Phase')

plt.figure(dpi = 150)
plt.imshow(pyh.phase(relativeField), cmap = 'twilight')
plt.title('Relative Phase')

plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Intensity')

DIC = pyh.synthetic_DIC(reconField, shearAngle = 0)
plt.figure(dpi = 150)
plt.imshow(DIC, cmap='gray')
plt.title('Synthetic DIC')

phaseGrad = pyh.phase_gradient(correctedField)
plt.figure(dpi = 150)
plt.imshow(phaseGrad, cmap='gray')
plt.title('Phase Gradient')