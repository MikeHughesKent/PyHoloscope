# -*- coding: utf-8 -*-
"""
Tests off-axis holography functionality of PyHoloscope using low level functions.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt

import time
import math

from pathlib import Path

import context                    # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = .6e-6

# Load images
hologram = pyh.load_image(Path('test data/tissue_paper_oa.tif'))
background = pyh.load_image(Path('test data/tissue_paper_oa_background.tif'))

# Determine Modulation
cropCentre = pyh.off_axis_find_mod(background)
cropRadius = pyh.off_axis_find_crop_radius(background)

# Remove modulation    
reconField = pyh.off_axis_demod(hologram.astype(float), cropCentre, cropRadius)
backgroundField = pyh.off_axis_demod(background.astype(float), cropCentre, cropRadius)

# Apply background correction and phase offset correction
correctedField = pyh.relative_phase(reconField, backgroundField)

# Display results
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Amplitude, no mask')
    
plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase, no mask')




""" Circular Mask """

# Remove modulation    
mask = pyh.circ_window( (cropRadius[0] * 2, cropRadius[1] * 2), cropRadius)
reconField = pyh.off_axis_demod(hologram.astype(float), cropCentre, cropRadius, mask = mask)
backgroundField = pyh.off_axis_demod(background.astype(float), cropCentre, cropRadius)

# Apply background correction and phase offset correction
correctedField = pyh.relative_phase(reconField, backgroundField)

# Display results
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Amplitude, circ mask')
    
plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase, circ mask')


""" Cosine Mask """


# Remove modulation    
mask = pyh.circ_cosine_window( (cropRadius[0] * 2, cropRadius[1] * 2), cropRadius, 10)

reconField = pyh.off_axis_demod(hologram.astype(float), cropCentre, cropRadius, mask = mask)
backgroundField = pyh.off_axis_demod(background.astype(float), cropCentre, cropRadius)

# Apply background correction and phase offset correction
correctedField = pyh.relative_phase(reconField, backgroundField)

# Display results
plt.figure(dpi = 150)
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')
plt.title('Amplitude, cos mask')
    
plt.figure(dpi = 150)
plt.imshow(pyh.phase(reconField), cmap = 'twilight')
plt.title('Phase, cos mask')
