# -*- coding: utf-8 -*-
"""
Tests inline holography depth stack functionality of PyHoloscope

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt

import context                    # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0127

# Load images
hologram = pyh.load_image("test data\\inline_example_holo.tif")
background = pyh.load_image("test data\\inline_example_back.tif")

# Set up PyHoloscope
holo = pyh.Holo(pyh.INLINE_MODE, wavelength, pixelSize)
holo.set_background(background)
holo.set_depth(depth)

depthRange = [0, 0.02]
nDepths = 10
stack = holo.depth_stack(hologram, depthRange, nDepths)

# Display results
plt.figure(dpi = 150)
plt.imshow(stack.get_depth_intensity(depth), cmap = 'gray')
plt.title('Refocused Hologram')