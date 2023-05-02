# -*- coding: utf-8 -*-
"""
Tests inline holography functionality of PyHoloscope

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import time

import context                    # Relative paths

import pyholoscope as pyh

import numpy as np

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0127

# Load images
hologram = pyh.load_image("test data\\inline_example_holo.tif")
background = pyh.load_image("test data\\inline_example_back.tif")
gridSize = np.shape(hologram)[0]

propagator = pyh.propagator(gridSize, wavelength, pixelSize, depth)


# Set up PyHoloscope
#holo = pyh.Holo(mode = pyh.INLINE_MODE, 
#                wavelength = wavelength, 
#                pixelSize= pixelSize,
#                background = background,
#                depth = depth)


propagator = pyh.propagator(gridSize, wavelength, pixelSize, depth, realImage = True)

# t1 = time.perf_counter()
# holo.update_propagator(hologram)    # To make timing below just for refocusing
#                                     # and not for propagator generation
# print(f"Inline propgator generation time: {round((time.perf_counter() - t1) * 1000)} ms")


# # Refocus
# t1 = time.perf_counter()
recon = pyh.refocus(hologram - background, propagator, realImage = True)
# print(f"Inline refocusing time: {round((time.perf_counter() - t1) * 1000)} ms")

# # Extract anmpltide
amp = pyh.amplitude(recon)

# Display results
plt.figure(dpi = 150)
plt.imshow(amp, cmap = 'gray')
plt.title('Refocused Hologram')