# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography with numerical refocusing
functionality of PyHoloscope using Holo class.

"""

import context

from matplotlib import pyplot as plt

from pathlib import Path

import time
import math

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixel_size = 1e-6
depth = 925e-6

# Load images
hologram = pyh.load_image(Path("test data/paramecium_oa_oof.tif"))
background = pyh.load_image(Path("test data/paramecium_oa_oof_background.tif"))

# Create object
holo = pyh.Holo(
    mode=pyh.OFF_AXIS,
    wavelength=wavelength,
    pixel_size=pixel_size,
    background=background,
    auto_window=False,
    relative_phase=True,
    refocus=True,
    geometry="point",
    depth=depth,
)

holo.calib_off_axis()  # Finds modulation frequency and background field

t1 = time.perf_counter()
recon_field = holo.process(hologram)
print(
    f"Demodulation and refocusing time: {round((time.perf_counter() - t1) * 1000)} ms"
)

# Display intensity and phase
plt.figure(dpi=150)
plt.imshow(pyh.amplitude(recon_field), cmap="gray", interpolation="none")
plt.title("Intensity")

plt.figure(dpi=150)
plt.imshow(pyh.phase(recon_field), cmap="twilight", interpolation="none")
plt.title("Phase")

DIC = pyh.synthetic_DIC(recon_field, shear_angle=0)
plt.figure(dpi=150)
plt.imshow(DIC, cmap="gray", interpolation="none")
plt.title("Synthetic DIC")

phase_grad = pyh.phase_gradient(recon_field)
plt.figure(dpi=150)
plt.imshow(phase_grad, cmap="gray")
plt.title("Phase Gradient")

# Unwrap phase
phase_unwrapped = pyh.phase_unwrap(pyh.phase(recon_field))
plt.figure(dpi=150)
plt.imshow(phase_unwrapped)
plt.title("Phase Unwrapped")

# Remove image phase tilt
tilt = pyh.obtain_tilt(phase_unwrapped)
phase_untilted = phase_unwrapped - tilt

plt.figure(dpi=150)
plt.imshow(phase_untilted)
plt.title("Tilt Removed")
