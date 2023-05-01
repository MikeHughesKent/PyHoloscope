# -*- coding: utf-8 -*-
"""
Minimal example of how to use inline holography functionality of PyHoloscope.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt

import context                    # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0127

# Load images
hologram = pyh.load_image("..\\test\\test data\\inline_example_holo.tif")
background = pyh.load_image("..\\test\\test data\\inline_example_back.tif")

# Set up PyHoloscope
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                background = background,
                depth = depth)

# Refocus
recon = holo.process(hologram)

# Extract anmpltide
amp = pyh.amplitude(recon)

# Display results
plt.figure(dpi = 150)
plt.imshow(amp, cmap = 'gray')
plt.title('Refocused Hologram')