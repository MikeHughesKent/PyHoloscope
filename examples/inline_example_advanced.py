# -*- coding: utf-8 -*-
"""
More detailed example of how to use inline holography functionality of 
PyHoloscope.

See inline_example.py for a more minimal example.

This example loads an inline hologram and a background image (i.e. with the 
sample removed).

The images are loaded using the PyHoloscope 'load_image' function. 
Alternatively you can load these in using any method that results in them 
being stored in a 2D numpy array.

We instantiate a 'Holo' object and then pass in the system parameters and 
various options.

We call the 'process' method of 'Holo' to refocus the hologram. If you have 
a GPU and CuPy is installed the GPU will be used, otherwise it will revert to 
CPU.

We then add normalisation, background subtraction and windowing.

Finally we use the 'amplitude' function to extract the amplitude of the 
refocused image for display.

"""
from time import perf_counter as timer
from matplotlib import pyplot as plt

import context                    # Loads relative paths

import pyholoscope as pyh

from pathlib import Path

# Load images
holoFile = Path('../test/integration_tests/test data/inline_example_holo.tif')
backFile = Path('../test/integration_tests/test data/inline_example_back.tif')

hologram = pyh.load_image(holoFile)
backHologram = pyh.load_image(backFile)


""" Simple processing """
# Create an instance of the Holo class with bare minimum parameters. There is no
# background correction or normalisation
holo = pyh.Holo(mode = pyh.INLINE,             # For inline holography
                wavelength = 630e-9,           # Light wavelength, m
                pixelSize = 1e-6,              # Hologram physical pixel size, m
                depth = 0.013)                # Distance to refocus, m


# We call this here, but this is optional, otherwise
# the propagator will be created the first time we call 'process'.
holo.update_propagator(hologram)

# Refocus
recon = holo.process(hologram)



""" With normalisation"""
# We now add normalisation, we could have done this when we created the 
# Holo object, by passing in normlise = backHologram, but we can also add this 
# in later as follows:
holo.set_normalise(backHologram)    

# Refocus
reconNorm = holo.process(hologram)



""" With background and normalisation """
# We now add background subtraction, we could have done this is when we created 
# the Holo object, by passing in background = backHologram, but we can also add 
# this in later as follows:
holo.set_background(backHologram)    

# Refocus
reconNormBack = holo.process(hologram)



""" With background and normalisation and window """
# We now add a cosine window to reduce edge artefacts, we could have done this is when 
# we created the Holo object, by passing in autoWindow = True, but we can also add this 
# in later as follows:
holo.set_auto_window(True)

# By defualt the skin thickness (distance over which the window smoothly 
# changes from transparent) to opaque) is 10 pixels, but we can set a different 
# value
holo.set_window_thickness(20)

# We pre-compute the window, this is optional and would be done the next time we call
# process. We have to pass in either the background or the hologram so that holo 
# knows how large to make the window.
holo.update_auto_window(backHologram)

# Refocus
reconNormBackWindow = holo.process(hologram)



""" Refocusing to a different depth """
# We now refocus the hologram to a different depth. We change the refocus depth using:
holo.set_depth(0.01)

# We could call update_propagator() here, but we don't have to as PyHoloscope will
# realise the depth has changed and regenerate the propagator when we called process.
# This process will take a little longer the first time we call it since we are 
# generation the propagator.

reconNormBackWindow2 = holo.process(hologram)



""" Display results """
plt.figure(dpi = 150); plt.title('Raw Hologram')
plt.imshow(hologram, cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram')
plt.imshow(pyh.amplitude(recon), cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram with \n Normalisation')
plt.imshow(pyh.amplitude(reconNorm), cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram with Background \n and Normalisation')
plt.imshow(pyh.amplitude(reconNormBack), cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram with Background, \nNormalisation and Windowing')
plt.imshow(pyh.amplitude(reconNormBackWindow), cmap = 'gray')

plt.figure(dpi = 150); plt.title('Refocused Hologram (Wrong Depth) with Background, \nNormalisation and Windowing')
plt.imshow(pyh.amplitude(reconNormBackWindow2), cmap = 'gray')