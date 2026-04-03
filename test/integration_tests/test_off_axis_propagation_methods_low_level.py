# -*- coding: utf-8 -*-
"""
Tests off-axis low-level refocusing with angular spectrum and Fresnel propagation.

"""

import context

from matplotlib import pyplot as plt
from pathlib import Path

import time
import numpy as np

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixel_size = 1e-6
depth = 0.00117

# Load images
hologram = pyh.load_image(Path("test data/paramecium_oa_oof.tif"))
background = pyh.load_image(Path("test data/paramecium_oa_oof_background.tif"))

# Determine modulation
crop_centre = pyh.off_axis_find_mod(background)
crop_radius = pyh.off_axis_find_crop_radius(background)

# Demodulate and apply background correction
recon_field = pyh.off_axis_demod(hologram, crop_centre, crop_radius)
background_field = pyh.off_axis_demod(background, crop_centre, crop_radius)
corrected_field = pyh.relative_phase(recon_field, background_field)

adjusted_pixel_size = pixel_size / (crop_radius[0] * 2) * np.shape(hologram)[1]
prop_size = (crop_radius[0] * 2, crop_radius[1] * 2)

# Warm up numba JIT for fairer timing
_ = pyh.propagator(
    prop_size,
    wavelength,
    adjusted_pixel_size,
    depth,
    propagation_method="angular_spectrum",
)

# Angular spectrum
t1 = time.perf_counter()
prop_angular = pyh.propagator(
    prop_size,
    wavelength,
    adjusted_pixel_size,
    depth,
    propagation_method="angular_spectrum",
)
print(
    f"Off-axis angular spectrum propagator generation time: {round((time.perf_counter() - t1) * 1000)} ms"
)

t1 = time.perf_counter()
refocused_angular = pyh.refocus(corrected_field, prop_angular)
print(
    f"Off-axis angular spectrum refocusing time: {round((time.perf_counter() - t1) * 1000)} ms"
)

# Fresnel
t1 = time.perf_counter()
prop_fresnel = pyh.propagator(
    prop_size,
    wavelength,
    adjusted_pixel_size,
    depth,
    propagation_method="fresnel",
)
print(
    f"Off-axis fresnel propagator generation time: {round((time.perf_counter() - t1) * 1000)} ms"
)

t1 = time.perf_counter()
refocused_fresnel = pyh.refocus(corrected_field, prop_fresnel)
print(f"Off-axis fresnel refocusing time: {round((time.perf_counter() - t1) * 1000)} ms")

# Display results
plt.figure(dpi=150)
plt.imshow(pyh.amplitude(refocused_angular), cmap="gray", interpolation="none")
plt.title("Off-axis Refocus: Angular Spectrum")

plt.figure(dpi=150)
plt.imshow(pyh.phase(refocused_angular), cmap="twilight", interpolation="none")
plt.title("Off-axis Phase: Angular Spectrum")

plt.figure(dpi=150)
plt.imshow(pyh.amplitude(refocused_fresnel), cmap="gray", interpolation="none")
plt.title("Off-axis Refocus: Fresnel")

plt.figure(dpi=150)
plt.imshow(pyh.phase(refocused_fresnel), cmap="twilight", interpolation="none")
plt.title("Off-axis Phase: Fresnel")

plt.figure(dpi=150)
plt.imshow(
    pyh.amplitude(refocused_angular - refocused_fresnel),
    cmap="gray",
    interpolation="none",
)
plt.title("Off-axis Amplitude Difference: Angular Spectrum - Fresnel")
