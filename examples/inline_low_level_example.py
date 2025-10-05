# -*- coding: utf-8 -*-
"""
Minimal example of how to use low level inline holography functionality of
PyHoloscope.

This example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.

Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

"""

from matplotlib import pyplot as plt
from pathlib import Path

import context  # Loads relative paths

import pyholoscope as pyh


# Load hologram and background images
holoFile = Path("../test/integration_tests/test data/inline_example_holo.tif")
backFile = Path("../test/integration_tests/test data/inline_example_back.tif")

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)


# Create the angular spectrum propagator to refocus to required depth
prop = pyh.propagator(
    wavelength=630e-9, pixel_size=1e-6, depth=0.0130, grid_size=hologram
)

# Refocus
recon = pyh.refocus(hologram, prop)


# Generate a spatial window
window = pyh.square_cosine_window(hologram, skin_thickness=10)

# Refocus with background subtraction, normalisation and windowing
recon_adv = pyh.refocus(
    hologram, prop, background=background, normalise=background, window=window
)


# Extract amplitude
amp = pyh.amp(recon)
amp_adv = pyh.amp(recon_adv)


# Display results
plt.figure(dpi=150)
plt.title("Raw Hologram")
plt.imshow(hologram, cmap="gray")

plt.figure(dpi=150)
plt.title("Refocused Hologram (amp)")
plt.imshow(amp, cmap="gray")

plt.figure(dpi=150)
plt.title("Refocused Hologram with back, norm and window (amp)")
plt.imshow(amp, cmap="gray")

plt.figure(dpi=150)
plt.title("Refocused Hologram (inverted)")
plt.imshow(pyh.invert(amp), cmap="gray")
