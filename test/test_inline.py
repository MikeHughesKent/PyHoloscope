# -*- coding: utf-8 -*-
"""
Tests inline holography functionality of PyHoloscope

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import time
from pathlib import Path


import context                    # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0127

# Load images
hologram = pyh.load_image(Path('test data/inline_example_holo.tif'))
background = pyh.load_image(Path('test data/inline_example_back.tif'))

# Set up PyHoloscope
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                background = background,
                normalise = background,
                depth = depth,
                autoWindow = True)


t1 = time.perf_counter()
holo.update_propagator(hologram)         # To make timing below just for refocusing
print(f"Propagator generation time: {round((time.perf_counter() - t1) * 1000)} ms")


t1 = time.perf_counter()
holo.update_auto_window(hologram)        # To make timing below just for propagator/window generation
print(f"Window generation time: {round((time.perf_counter() - t1) * 1000)} ms")


# Refocus
t1 = time.perf_counter()
recon = holo.process(hologram)
print(f"Inline refocusing time with window: {round((time.perf_counter() - t1) * 1000)} ms")
plt.figure(dpi = 150); plt.imshow(pyh.amplitude(recon), cmap = 'gray'); plt.title('Refocused Hologram, pre-window')


# With post refocusing window applied
holo.set_post_window(True)
t1 = time.perf_counter()
recon = holo.process(hologram)
print(f"Inline refocusing time with post window: {round((time.perf_counter() - t1) * 1000)} ms")
plt.figure(dpi = 150); plt.imshow(pyh.amplitude(recon), cmap = 'gray'); plt.title('Refocused Hologram, pre and post window')


# Without window at all
holo.set_auto_window(False)
holo.clear_window()
t1 = time.perf_counter()
recon = holo.process(hologram)
print(f"Inline refocusing time with no window: {round((time.perf_counter() - t1) * 1000)} ms")
plt.figure(dpi = 150); plt.imshow(pyh.amplitude(recon), cmap = 'gray'); plt.title('Refocused Hologram, no window')


# No window or background
holo.clear_background()
holo.clear_normalise()
t1 = time.perf_counter()
recon = holo.process(hologram)
print(f"Inline refocusing time with no background or normalisation: {round((time.perf_counter() - t1) * 1000)} ms")
plt.figure(dpi = 150); plt.imshow(pyh.amplitude(recon), cmap = 'gray'); plt.title('Refocused Hologram, no background or norm')