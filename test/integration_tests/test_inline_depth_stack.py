# -*- coding: utf-8 -*-
"""
Tests inline holography depth stack functionality of PyHoloscope.

"""

from matplotlib import pyplot as plt
import time
from pathlib import Path

import context  # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 633e-9
pixel_size = 1e-6
depth = 0.0129

# Load images
hologram = pyh.load_image(Path("test data/inline_example_holo.tif"))
background = pyh.load_image(Path("test data/inline_example_back.tif"))

# Set up PyHoloscope
holo = pyh.Holo(
    pyh.INLINE_MODE,
    wavelength=wavelength,
    pixel_size=pixel_size,
    background=background,
    depth=depth,
)

# Range for depth stack
depth_range = [0, 0.02]
num_depths = 50


# Build depth stack
t1 = time.perf_counter()
stack = holo.depth_stack(hologram, depth_range, num_depths)
print("Time to generate stack (s): ", round(time.perf_counter() - t1, 2))

# Display correct fcous image
plt.figure(dpi=150)
plt.imshow(pyh.amplitude(holo.process(hologram)), cmap="gray")
plt.title("Refocused Hologram")

# Display results (focus depth)
plt.figure(dpi=150)
plt.imshow(stack.get_depth_intensity(depth), cmap="gray")
plt.title("Refocused Hologram from Stack (from depth)")

# Display results (should be same as above)
plt.figure(dpi=150)
plt.imshow(stack.get_index_intensity(32), cmap="gray")
plt.title("Refocused Hologram from Stack (from idx)")
