# -*- coding: utf-8 -*-
"""
Minimal example of how to use off-axis holography functionality of PyHoloscope
with numerical refocusing.

This example loads an off-axis hologram and a background image (i.e. with the 
sample removed).

The images are loaded using the PyHoloscope 'load_image' function. 
Alternatively you can load these in using any method that results in them 
being stored in a 2D numpy array.

We instantiate a 'Holo' object and pass in the system parameters and some 
options, including a flag to tell it to refocus to a particular depth.

We call the 'process' method of 'Holo' to demodulate the hologram to recover 
the phase and numerically refocus.
 
If you have a GPU and CuPy is installed the GPU will be used, otherwise it 
will revert to CPU.

We use the 'amplitude' and 'phase' functions to extract the amplitude 
and phase of the complex demodulated image. 

"""


from matplotlib import pyplot as plt
from pathlib import Path

import context         # Paths

import pyholoscope as pyh


# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
refocusDepth = -0.0012

# Load images
# Load images
holoFile = Path('../test/integration_tests/test data/paramecium_oa_oof.tif')
backFile = Path('../test/integration_tests/test data/paramecium_oa_oof_background.tif')

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)


# Create Holo object
holo = pyh.Holo(mode = pyh.OFF_AXIS, 
                wavelength = wavelength,
                precision = 'single',
                pixelSize = pixelSize,
                background = background,    # For correcting background phase
                relativePhase = True,       # We will remove the background phase
                refocus = True,             # We will numerically refocus
                depth = refocusDepth)       # Refocus distance in m

holo.calib_off_axis()                       # Finds modulation frequency and 
                                            # pre-computes background phase


# In a single step we remove the off-axis modulation and refocus
reconField = holo.process(hologram)


# Unwrap phase
phaseUnwrapped = pyh.phase_unwrap(pyh.phase(reconField))


# Detect a global tilt in the phase and remove it (note we must supply the
# unwrapped phase here)
tilt = pyh.obtain_tilt(phaseUnwrapped)
phaseUntilted = pyh.relative_phase(phaseUnwrapped, tilt)


""" Display results """
plt.figure(dpi = 150); plt.title('Intensity')
plt.imshow(pyh.amplitude(reconField), cmap = 'gray')

plt.figure(dpi = 150); plt.title('Phase')
plt.imshow(pyh.phase(reconField), cmap = 'twilight', interpolation='none')

