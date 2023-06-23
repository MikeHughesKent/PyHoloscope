# -*- coding: utf-8 -*-
"""
Minimal example of how to use inline holography functionality of PyHoloscope.

@author: Mike Hughes, Applied Optics Group, University of Kent

This example loads an inline hologram and a background image (i.e. with the sample removed).

The images are loaded using the PyHoloscope 'load_image' function. Altneratively you
can load these in using any method that results in them being stored in a 2D numpy array.

We instantiate a 'Holo' object and pass in the system parameters and some options.

We then use the 'update_propagator' and 'update_auto_window' methods of 'Holo' 
to pre-compute the angular spectrum propagator and the spatial window. If we 
don't do this they will be created the first time we call 'process'. If you have
the numba package installed this will be used to speed up propagator generation.

We call the 'process' method of 'Holo' to refocus the hologram. If you have 
a GPU and Cupy is installed the GPU will be used, otherwise it will revert to CPU.

Finally we use the 'amplitude' function to extract the amplitude of the refocused
image for display.

"""
from time import perf_counter as timer
from matplotlib import pyplot as plt

import context                    # Loads relative paths

import pyholoscope as pyh

from pathlib import Path

# Load images
holoFile = Path('../test/test data/inline_example_holo.tif')
backFile = Path('../test/test data/inline_example_back.tif')

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)

# Create an instance of the Holo class
holo = pyh.Holo(mode = pyh.INLINE,             # For inline holography
                windowThickness = 100,
                wavelength = 630e-9,           # Light wavelength, m
                pixelSize = 1e-6,              # Hologram physical pixel size, m
                background = background,       # To subtract the background
                normalise = background,        # To divide by the background
                autoWindow = True,             # Will result in a cosine window to smooth edges
                depth = 0.0130)                 # Distance to refocus, m


# We call these here, but this is optional, otherwise
# the propagator/window will be created the first time we call 'process'.
holo.update_propagator(hologram)
holo.update_auto_window(hologram)

# Refocus
startTime = timer()
recon = holo.process(hologram)
print(f"Numerical refocusing took {round((timer() - startTime) * 1000)} ms.")

# Extract amplitude and magnitude
amp = pyh.amplitude(recon)
mag = pyh.magnitude(recon)

# Display results
plt.figure(dpi = 150); plt.title('Raw Hologram')
plt.imshow(hologram, cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram (amp)')
plt.imshow(amp, cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram (mag, inverted)')
plt.imshow(pyh.invert(mag), cmap = 'gray')