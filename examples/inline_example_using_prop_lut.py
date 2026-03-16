# -*- coding: utf-8 -*-
"""
Example of how to use propagator Look up Table (LUT) duing
inline holography with PyHoloscope.

This example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.

Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

We instantiate a 'Holo' object and pass in the system parameters and some
options.

We then create a propagator LUT and set PyHoloscope to use the table for
refocusing.

We call the 'process' method of 'Holo' to refocus the hologram. If you have
a GPU and CuPy is installed the GPU will be used, otherwise it will revert to
CPU.

Finally we use the 'amplitude' function to extract the amplitude of the
refocused image for display.

"""

import time

from matplotlib import pyplot as plt
from pathlib import Path

import context  # Loads relative paths

import pyholoscope as pyh


# Load hologram and background images
holoFile = Path("../test/integration_tests/test data/inline_example_holo.tif")
backFile = Path("../test/integration_tests/test data/inline_example_back.tif")

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)


# Create an instance of the Holo class
holo = pyh.Holo(
    mode=pyh.INLINE,  # For inline holography
    wavelength=630e-9,  # Light wavelength, m
    pixel_size=1e-6,  # Hologram physical pixel size, m
    background=background,  # To subtract the background
    depth=0.0130,  # Distance to refocus, m
    use_prop_lut=True,
)


# Create a propagator LUT
depth_range = (0, 0.2)
num_depths = 200
holo.make_propagator_LUT(hologram, depth_range, num_depths)

print(f"Target depth: {holo.depth}")
print(
    f"Closest depth in LUT: {round(holo.propagator_lut.depths[holo.propagator_lut.closest_index(holo.depth)], 4)}"
)

# Refocus and extract amplitude
t1 = time.perf_counter()
recon_lut = holo.process(hologram)
print(f"LUT recon time: {round(time.perf_counter() - t1, 3)}")
amp_lut = pyh.amplitude(recon_lut)

# Adjust Holo class to not use LUT and reconstruct
holo.use_prop_lut = False
t1 = time.perf_counter()
recon = holo.process(hologram)
print(f"Standard recon time: {round(time.perf_counter() - t1, 3)}")
amp = pyh.amplitude(recon)

# Display results
plt.figure(dpi=150)
plt.title("Standard Recon")
plt.imshow(amp, cmap="gray")

plt.figure(dpi=150)
plt.title("Recon using LUT")
plt.imshow(amp_lut, cmap="gray")
