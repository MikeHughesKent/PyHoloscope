# -*- coding: utf-8 -*-
"""
Tests inline holography refocusing with angular spectrum and Fresnel propagation.

"""

from matplotlib import pyplot as plt
import time
from pathlib import Path

import context  # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 633e-9  # m
pixel_size = 1e-6  # m
depth = 0.0129  # m

# Load images
hologram = pyh.load_image(Path("test data/inline_example_holo.tif"))
background = pyh.load_image(Path("test data/inline_example_back.tif"))

# Pre-process hologram once so timings below are primarily propagator/refocus
hologram_pre = pyh.pre_process(hologram, background=background, normalise=background)

# Run once to force numba JIT compile
prop_angular = pyh.propagator(
    hologram_pre.shape,
    wavelength,
    pixel_size,
    depth,
    propagation_method="angular_spectrum",
)

# Angular spectrum refocusing
t1 = time.perf_counter()
prop_angular = pyh.propagator(
    hologram_pre.shape,
    wavelength,
    pixel_size,
    depth,
    propagation_method="angular_spectrum",
)
print(
    f"Angular spectrum propagator generation time: {round((time.perf_counter() - t1) * 1000)} ms"
)

t1 = time.perf_counter()
recon_angular = pyh.refocus(hologram_pre, prop_angular)
print(
    f"Angular spectrum refocusing time: {round((time.perf_counter() - t1) * 1000)} ms"
)

# Fresnel refocusing
t1 = time.perf_counter()
prop_fresnel = pyh.propagator(
    hologram_pre.shape,
    wavelength,
    pixel_size,
    depth,
    propagation_method="fresnel",
)
print(
    f"Fresnel propagator generation time: {round((time.perf_counter() - t1) * 1000)} ms"
)

t1 = time.perf_counter()
recon_fresnel = pyh.refocus(hologram_pre, prop_fresnel)
print(f"Fresnel refocusing time: {round((time.perf_counter() - t1) * 1000)} ms")

# Display both reconstructions and their absolute difference
plt.figure(dpi=150)
plt.imshow(pyh.amplitude(recon_angular), cmap="gray")
plt.title("Inline Refocus: Angular Spectrum")

plt.figure(dpi=150)
plt.imshow(pyh.amplitude(recon_fresnel), cmap="gray")
plt.title("Inline Refocus: Fresnel")

plt.figure(dpi=150)
plt.imshow(
    pyh.amplitude(recon_angular - recon_fresnel),
    cmap="gray",
)
plt.title("Amplitude Difference: Angular Spectrum - Fresnel")
