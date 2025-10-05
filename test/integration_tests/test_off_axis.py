# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography functionality of PyHoloscope

"""

import time

from matplotlib import pyplot as plt

import numpy as np

import context  # Load paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 0.3e-6

# Load images
hologram = pyh.load_image("test data\\tissue_paper_oa.tif")
background = pyh.load_image("test data\\tissue_paper_oa_background.tif")

# Create the Holo object that will be used for demodulation
holo = pyh.Holo(
    pyh.OFF_AXIS,
    wavelength=wavelength,
    pixel_size=pixelSize,
    background=background,
    crop_mask=pyh.Holo.CIRCLE_COSINE,
    crop_window_skin_thickness=10,
    relative_phase=True,
)

# Find modulation frequency and generate background and normalisation fields
# and create window
holo.calib_off_axis()

# Remove modulation
t1 = time.perf_counter()
recon_field = holo.process(hologram)
print(f"Off-axis demodulation time: {round((time.perf_counter() - t1) * 1000)} ms")

# Display results

plt.figure(dpi=150)
plt.imshow(hologram, cmap="gray")
plt.title("Hologram")

plt.figure(dpi=150)
plt.imshow(np.abs(recon_field), cmap="gray")
plt.title("Amplitude")

plt.figure(dpi=150)
plt.imshow(np.angle(recon_field), cmap="twilight")
plt.title("Phase")

DIC = pyh.synthetic_DIC(recon_field, shear_angle=0)
plt.figure(dpi=150)
plt.imshow(DIC, cmap="gray")
plt.title("Synthetic DIC")

phase_grad = pyh.phase_gradient(recon_field)
plt.figure(dpi=150)
plt.imshow(phase_grad, cmap="gray")
plt.title("Phase Gradient")
