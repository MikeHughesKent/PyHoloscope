# -*- coding: utf-8 -*-
"""
Example of how to use off-axis holography functionality of PyHoloscope with phase
processing and visualisation. 

@author: Mike Hughes, Applied Optics Group, University of Kent

This example loads an off-axis hologram and a background image (i.e. with the 
sample removed).

The images are loaded using the PyHoloscope 'load_image' function. 
Altneratively you can load these in using any method that results in them 
being stored in a 2D numpy array.

We instantiate a 'Holo' object and pass in the system parameters and some 
options.

We call the 'process' method of 'Holo' to demodulate the hologram to recover 
the phase and to numerically refocus.
 
If you have a GPU and Cupy is installed the GPU will be used, otherwise it 
will revert to CPU.

We use the 'amplitude' and 'phase' functions to extract the amplitude 
and phase of the complex demodulated image. 

We then use some low-level functions to process and display the phase in 
different ways.
"""

import context         # Paths

from time import perf_counter as timer
from matplotlib import pyplot as plt

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6

# Load images
hologram = pyh.load_image("..\\test\\test data\\paramecium_oa_oof.tif")
background = pyh.load_image("..\\test\\test data\\paramecium_oa_oof_background.tif")

# Create Holo object
holo = pyh.Holo(mode = pyh.OFF_AXIS, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                background = background,    # For correcting background phase
                relativePhase = True,       # We will remove the background phase
                refocus = True,             # We will numerically refocus
                depth = -0.0012)            # Refocus distance in m

holo.calib_off_axis()                       # Finds modulation frequency and 
                                            # pre-computes background phase


# We call this here, but this is optional, otherwise
# the propagator will be created the first time we call 'process'.
holo.update_propagator(hologram)

# In a single step we remove the off-axis modulation and refocus
startTime = timer()
reconField = holo.process(hologram)
print(f"Off-axis demodulation and refocusing took {round((timer() - startTime) * 1000)} ms.")

# We had set relativePhase = True which means that we subtracted the phase
# from the background image. The alternative is to call relative_phase to do
# this manually.

# The output from holo.process is complex. If we extract the phase it will be 
# wrapped
phase = pyh.phase(reconField)

# We can perform 2D phase unwrapping using
phaseUnwrapped = pyh.phase_unwrap(phase)

# It is sometimes the case that there is still a tilt in the phase after
# we removed the background. For example, maybe the cover slip is slightly
# angle. We can detect this using:
tilt = pyh.obtain_tilt(phaseUnwrapped)

# And remove the tilted phase:
phaseUntilted = pyh.relative_phase(phaseUnwrapped, tilt)

# We can create a DIC style image from the field - we have to provde
# the field as the DIC relies on both amplitude and phase
DIC = pyh.synthetic_DIC(reconField)

# We can also create a phase gradient image. We can do this either from the 
# raw field or the raw phase (results will be the same) or from any of the 
# processed phase maps (results will tend to be similar but not identical). 
phaseGrad = pyh.phase_gradient(phaseUntilted)
phaseGradRaw = pyh.phase_gradient(reconField)


""" Display results """
plt.figure(dpi = 150); plt.title('Raw Phase')
plt.imshow(phase, cmap = 'twilight', interpolation='none')

plt.figure(dpi = 150); plt.title('Unwrapped Phase')
plt.imshow(phaseUnwrapped, cmap = 'twilight', interpolation='none')

plt.figure(dpi = 150); plt.title('Unwrapped and Untilted Phase')
plt.imshow(phaseUntilted, cmap = 'twilight', interpolation='none')

plt.figure(dpi = 150); plt.title('Synthetic DIC from Raw Field')
plt.imshow(DIC, cmap='gray', interpolation='none')

plt.figure(dpi = 150); plt.title('Phase Gradient from Processed Phase')
plt.imshow(phaseGrad, cmap='gray', interpolation='none')

plt.figure(dpi = 150); plt.title('Phase Gradient from Raw Field')
plt.imshow(phaseGradRaw, cmap = 'gray', interpolation='none')
