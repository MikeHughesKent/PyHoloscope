# -*- coding: utf-8 -*-
"""
More detialed example of how to use inline holography functionality of PyHoloscope.

See inline_example.py for a more minimal example.

@author: Mike Hughes, Applied Optics Group, University of Kent

This example loads an inline hologram and a background image (i.e. with the sample removed).

The images are loaded using the PyHoloscope 'load_image' function. Altneratively you
can load these in using any method that results in them being stored in a 2D numpy array.

We instantiate a 'Holo' object and then pass in the system parameters and various options.

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
backHologram = pyh.load_image(backFile)


""" Simple processing """
# Create an instance of the Holo class with bare minimum parameters. There is no
# background correction, normalisation or depth
holo = pyh.Holo(mode = pyh.INLINE,             # For inline holography
                wavelength = 630e-9,           # Light wavelength, m
                pixelSize = 1e-6,              # Hologram physical pixel size, m
                depth = 0.0127)                # Distance to refocus, m


# We call these here, but this is optional, otherwise
# the propagator/window will be created the first time we call 'process'.
holo.update_propagator(hologram)
holo.update_auto_window(hologram)

# Refocus
startTime = timer()
recon = holo.process(hologram)
print(f"Numerical refocusing took {round((timer() - startTime) * 1000)} ms.")



""" With normalisation"""
# We now add normalisation, we could have done this is when we created the Holo object,
# by passing in normlise = backHologram, but we can also add this in later as follows:
holo.set_normalise(backHologram)    

# Refocus
startTime = timer()
reconNorm = holo.process(hologram)
print(f"Numerical refocusing with normalisation took {round((timer() - startTime) * 1000)} ms.")




""" With background and normalisation """
# We now add background subtraction, we could have done this is when we created the Holo object,
# by passing in background = backHologram, but we can also add this in later as follows:
holo.set_background(backHologram)    

# Refocus
startTime = timer()
reconNormBack = holo.process(hologram)
print(f"Numerical refocusing with normalisation and background took {round((timer() - startTime) * 1000)} ms.")






""" With background and normalisation and window """
# We now add a cosine window to reduce edge artefacts, we could have done this is when 
# we created the Holo object, by passing in autoWindow = True, but we can also add this 
# in later as follows:
holo.set_auto_window(True)

# By defualt the skin thickness (distance over which the window smoothly changes from transparent)
# to opaque) is 10 pixels, but we can set a different value
holo.set_window_thickness(20)

# We pre-compute the window, this is optional and would be done the next time we call
# process. We have to pass in either the background or the hologram so that holo 
# knows how large to make the window.
holo.update_auto_window(backHologram)

# Refocus
startTime = timer()
reconNormBackWindow = holo.process(hologram)
print(f"Numerical refocusing with normalisation, background and windowing took {round((timer() - startTime) * 1000)} ms.")




""" Refocusing to a different depth """
# We now refocus the hologram to a different depth. We change the refocus depth using:
holo.set_depth(0.01)

# We could call update_propagator() here, but we don't have to as PyHoloscope will
# realise the depth has changed and regenerate the propagator when we called process.
# This process will take a little longer the first tiem we call it since we are 
# generation the propagator.

startTime = timer()
reconNormBackWindow2 = holo.process(hologram)
print(f"Numerical refocusing with normalisation, background and windowing including propagator regeneration took {round((timer() - startTime) * 1000)} ms.")




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