# -*- coding: utf-8 -*-
"""
Tests off-axis holography functionality of PyHoloscope using low level functions.

"""

from matplotlib import pyplot as plt

import time
import math

from pathlib import Path

import context  # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixel_size = 0.6e-6

# Load images
hologram = pyh.load_image(Path("test data/tissue_paper_oa.tif"))
background = pyh.load_image(Path("test data/tissue_paper_oa_background.tif"))

# Determine Modulation
crop_centre = pyh.off_axis_find_mod(background)
crop_radius = pyh.off_axis_find_crop_radius(background)

# Remove modulation
recon_field = pyh.off_axis_demod(hologram.astype(float), crop_centre, crop_radius)
background_field = pyh.off_axis_demod(background.astype(float), crop_centre, crop_radius)

# Apply background correction and phase offset correction
corrected_field = pyh.relative_phase(recon_field, background_field)

# Display results
plt.figure(dpi=150)
plt.imshow(pyh.amp(recon_field), cmap="gray")
plt.title("Amplitude, no mask")

plt.figure(dpi=150)
plt.imshow(pyh.phase(recon_field), cmap="twilight")
plt.title("Phase, no mask")


""" Circular Mask """

# Remove modulation
mask = pyh.circ_window((crop_radius[0] * 2, crop_radius[1] * 2), crop_radius)
recon_field = pyh.off_axis_demod(
    hologram.astype(float), crop_centre, crop_radius, mask=mask
)
background_field = pyh.off_axis_demod(background.astype(float), crop_centre, crop_radius)

# Apply background correction and phase offset correction
corrected_field = pyh.relative_phase(recon_field, background_field)

# Display results
plt.figure(dpi=150)
plt.imshow(pyh.amp(recon_field), cmap="gray")
plt.title("Amplitude, circ mask")

plt.figure(dpi=150)
plt.imshow(pyh.phase(recon_field), cmap="twilight")
plt.title("Phase, circ mask")


""" Cosine Mask """


# Remove modulation
mask = pyh.circ_cosine_window((crop_radius[0] * 2, crop_radius[1] * 2), crop_radius, 10)

recon_field = pyh.off_axis_demod(
    hologram.astype(float), crop_centre, crop_radius, mask=mask
)
background_field = pyh.off_axis_demod(background.astype(float), crop_centre, crop_radius)

# Apply background correction and phase offset correction
corrected_field = pyh.relative_phase(recon_field, background_field)

# Display results
plt.figure(dpi=150)
plt.imshow(pyh.amplitude(recon_field), cmap="gray")
plt.title("Amplitude, cos mask")

plt.figure(dpi=150)
plt.imshow(pyh.phase(recon_field), cmap="twilight")
plt.title("Phase, cos mask")
