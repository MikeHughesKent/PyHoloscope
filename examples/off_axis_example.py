# -*- coding: utf-8 -*-
"""
Minimal example of how to use off-axis holography functionality of PyHoloscope.

@author: Mike Hughes, Applied Optics Group, University of Kent

This example loads an off-axis hologram and a background image (i.e. with the 
sample removed).

The images are loaded using the PyHoloscope 'load_image' function. 
Altneratively you can load these in using any method that results in them 
being stored in a 2D numpy array.

We instantiate a 'Holo' object and pass in the system parameters and some 
options.

We call the 'process' method of 'Holo' to demodulate the hologram to recover 
the phase.
 
If you have a GPU and Cupy is installed the GPU will be used, otherwise it 
will revert to CPU.

We use the 'amplitude' and 'phase' functions to extract the amplitude 
and phase of the complex demodulated image. 

"""

import context         # Paths
from time import perf_counter as timer

from matplotlib import pyplot as plt

from pathlib import Path

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = .3e-6


# Load images
holoFile = Path('../test/test data/tissue_paper_oa.tif')
backFile = Path('../test/test data/tissue_paper_oa_background.tif')

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)


# Create Holo object
holo = pyh.Holo(mode = pyh.OFF_AXIS,
                background = background,    # For correcting background phase
                relativePhase = True)       # We will remove the background phase
 
holo.calib_off_axis()                       # Finds modulation frequency and 
                                            # pre-computes background phase


# Remove the off-axis modulation and recover the phase
startTime = timer()
reconField = holo.process(hologram)
print(f"Off-axis demodulation took {round((timer() - startTime) * 1000)} ms.")


""" Display results """
plt.figure(dpi = 150); plt.title('Hologram')
plt.imshow(hologram, cmap = 'gray')

plt.figure(dpi = 150); plt.title('Intensity')
plt.imshow(pyh.amplitude(reconField), cmap = 'gray', interpolation='none')

plt.figure(dpi = 150); plt.title('Phase')
plt.imshow(pyh.phase(reconField), cmap = 'twilight', interpolation='none')

