# -*- coding: utf-8 -*-
"""
Example of how to use low level inline holography functionality of
PyHoloscope with a propagator look up table (LUT).

This example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.

Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

A propagator LUT is generated and the results of refocusing using a 
propagator from the LUT is compared with a standard reconstruction.

"""
import time

from matplotlib import pyplot as plt
from pathlib import Path

import context  # Loads relative paths

import pyholoscope as pyh


wavelength = 630e-9
pixel_size = 1e-6
depth = 0.0130

# Load hologram and background images

holoFile = Path("../test/integration_tests/test data/inline_example_holo.tif")
backFile = Path("../test/integration_tests/test data/inline_example_back.tif")

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)



# Create the angular spectrum propagator to refocus to required depth
t1 = time.perf_counter()
prop = pyh.propagator(
    wavelength=wavelength, pixel_size=pixel_size, depth=depth, grid_size=hologram
)
print(f"Time to generate a propagator: {round(time.perf_counter() - t1, 5)}")


# Refocus with background subtraction and normalisation
recon = pyh.refocus(
    hologram, prop, background=background, normalise=background,
)
amp = pyh.amp(recon)


# Create a LUT of angular spectrum propagators

depth_range = (0, 0.02)
num_depths = 200

prop_lut = pyh.PropLUT(hologram,
     wavelength,
     pixel_size,
     depth_range,
     num_depths,
     )

# Pull out propagator from LUT
t1 = time.perf_counter()
prop = prop_lut.propagator(depth)
print(f"Time to extract propagator from LUT: {round(time.perf_counter() - t1, 5)}")


# Refocus with background subtraction and normalisation
recon_lut = pyh.refocus(
    hologram, prop, background=background, normalise=background,
)
amp_lut = pyh.amp(recon_lut)


# Display results
plt.figure(dpi=150)
plt.title("Standard Recon")
plt.imshow(amp, cmap="gray")

plt.figure(dpi=150)
plt.title("Recon using LUT")
plt.imshow(amp_lut, cmap="gray")
