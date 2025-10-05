# -*- coding: utf-8 -*-
"""
Tests relative phase part of off axis holography functionality of PyHoloscope

"""

from matplotlib import pyplot as plt

import numpy as np
import time
from pathlib import Path

import context  # Load paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixel_size = 0.3e-6

# Load images
hologram = pyh.load_image(Path("test data/tissue_paper_oa.tif"))
background = pyh.load_image(Path("test data/tissue_paper_oa_background.tif"))

# Create object
holo = pyh.Holo(
    pyh.OFF_AXIS,
    wavelength=wavelength,
    pixel_size=pixel_size,
    background=background,
    relative_phase=False,
)


# Find modulation frequency
holo.calib_off_axis(background)

# Remove modulation
recon_field = holo.process(hologram)

recon_field_corrected = pyh.relative_phase_self(
    recon_field, roi=pyh.Roi(40, 40, 10, 10)
)


plt.figure(dpi=150)
plt.imshow(np.angle(recon_field), cmap="twilight")
plt.title("Phase uncorrected")

plt.figure(dpi=150)
plt.imshow(np.abs(recon_field), cmap="gray")
plt.title("Amplitude uncorrected")

plt.figure(dpi=150)
plt.imshow(np.angle(recon_field_corrected), cmap="twilight")
plt.title("Phase corrected")

plt.figure(dpi=150)
plt.imshow(np.abs(recon_field_corrected), cmap="gray")
plt.title("Amplitude corrected")
