# -*- coding: utf-8 -*-
"""
Tests inline Holo class refocusing with angular spectrum and Fresnel propagation.

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

# Set up PyHoloscope Holo class
holo = pyh.Holo(
    mode=pyh.INLINE_MODE,
    wavelength=wavelength,
    pixel_size=pixel_size,
    background=background,
    normalise=background,
    depth=depth,
    auto_window=True,
    propagation_method="angular_spectrum",
)

# Warm up each method once to avoid including one-time overhead
holo.propagation_method = "angular_spectrum"
holo.update_propagator(hologram)
_ = holo.process(hologram)

holo.propagation_method = "fresnel"
holo.update_propagator(hologram)
_ = holo.process(hologram)

# Angular spectrum
t1 = time.perf_counter()
holo.propagation_method = "angular_spectrum"
recon_angular = holo.process(hologram)
print(
    f"Inline Holo angular spectrum refocusing time: {round((time.perf_counter() - t1) * 1000)} ms"
)

# Fresnel
t1 = time.perf_counter()
holo.propagation_method = "fresnel"
recon_fresnel = holo.process(hologram)
print(
    f"Inline Holo fresnel refocusing time: {round((time.perf_counter() - t1) * 1000)} ms"
)

if recon_angular is None or recon_fresnel is None:
    raise Exception("Refocusing returned None for one or both propagation methods.")

# Display both reconstructions and their difference
plt.figure(dpi=150)
plt.imshow(pyh.amplitude(recon_angular), cmap="gray")
plt.title("Inline Holo Refocus: Angular Spectrum")

plt.figure(dpi=150)
plt.imshow(pyh.amplitude(recon_fresnel), cmap="gray")
plt.title("Inline Holo Refocus: Fresnel")

plt.figure(dpi=150)
plt.imshow(
    pyh.amplitude(recon_angular - recon_fresnel),
    cmap="gray",
)
plt.title("Inline Holo Amplitude Difference: Angular Spectrum - Fresnel")
