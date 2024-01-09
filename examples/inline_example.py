# -*- coding: utf-8 -*-
"""
Minimal example of how to use inline holography functionality of PyHoloscope.

This example loads an inline hologram and a background image (i.e. with the 
sample removed).

The images are loaded using the PyHoloscope 'load_image' function. 

Alternatively you can load these in using any method that results in them 
being stored in a 2D numpy array.

We instantiate a 'Holo' object and pass in the system parameters and some 
options.

We call the 'process' method of 'Holo' to refocus the hologram. If you have 
a GPU and Cupy is installed the GPU will be used, otherwise it will revert to 
CPU.

Finally we use the 'amplitude' function to extract the amplitude of the 
refocused image for display.

"""
from matplotlib import pyplot as plt
from pathlib import Path

import context                    # Loads relative paths

import pyholoscope as pyh


# Load hologram and background images
holoFile = Path('../test/test data/inline_example_holo.tif')
backFile = Path('../test/test data/inline_example_back.tif')

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)


# Create an instance of the Holo class
holo = pyh.Holo(mode = pyh.INLINE,             # For inline holography
                wavelength = 630e-9,           # Light wavelength, m
                pixelSize = 1e-6,              # Hologram physical pixel size, m
                background = background,       # To subtract the background
                depth = 0.0130)                # Distance to refocus, m

# Refocus
recon = holo.process(hologram)

# Extract amplitude and magnitude
amp = pyh.amplitude(recon)

# Display results
plt.figure(dpi = 150); plt.title('Raw Hologram')
plt.imshow(hologram, cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram (amp)')
plt.imshow(amp, cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram (inverted)')
plt.imshow(pyh.invert(amp), cmap = 'gray')