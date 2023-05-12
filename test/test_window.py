# -*- coding: utf-8 -*-
"""
Tests windowing functionality of PyHoloscope

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time
from pathlib import Path

import context                    # Relative paths

import pyholoscope as pyh


# Experimental Parameters
wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0127

# Load image
hologram = pyh.load_image(Path('test data/inline_example_holo.tif'))


# Check that autoWindow results in a window being created when we call update_auto_window
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,                
                depth = depth,
                autoWindow = True)

holo.update_auto_window(hologram)
assert np.shape(holo.window) == np.shape(hologram)


# Check that default window is the same as one created manually 
windowLowLevel = pyh.square_cosine_window(np.shape(hologram)[0], np.shape(hologram)[0] / 2, 10)
assert np.array_equal(windowLowLevel, holo.window) == True



# Check that autoWindow creates a window of the specified radius and defualt thickness
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,                
                depth = depth,
                autoWindow = True,
                windowRadius = 100)
holo.update_auto_window(hologram)

windowLowLevel = pyh.square_cosine_window(np.shape(hologram)[0], 100, 10)
assert np.array_equal(windowLowLevel, holo.window) == True



# Check that autoWindow creates a window of the specified radius and thickness
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,                
                depth = depth,
                autoWindow = True,
                windowRadius = 100,
                windowThickness = 20)
holo.update_auto_window(hologram)

windowLowLevel = pyh.square_cosine_window(np.shape(hologram)[0], 100, 20)
assert np.array_equal(windowLowLevel, holo.window) == True


# Check that the wrong window size gets resized with a warning displayed
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,                
                depth = depth)

windowLowLevel = pyh.square_cosine_window(int(np.shape(hologram)[0] / 2), 100, 20)
holo.set_window(windowLowLevel)

recon = holo.process(hologram)
windowLowLevel = pyh.square_cosine_window(int(np.shape(hologram)[0] / 2), 100, 20)
holo.set_window(windowLowLevel)
recon = holo.process(hologram)

# Check that if autoWindow not set, update_auto_window does not create a window
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,                
                depth = depth)

holo.update_auto_window(hologram)
assert holo.window is None


# Check we can set autoWindow using setter
holo.set_auto_window(True)
holo.update_auto_window(hologram)
assert np.shape(holo.window) == np.shape(hologram)


# Check that autoWindow results in a window being created when we call process
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                autoWindow = True)

recon = holo.process(hologram)
assert np.shape(holo.window) == np.shape(hologram)


# Check that autoWindow results in the correct size window being created when 
# we call process with downsampling
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                downsample = 2,
                autoWindow = True)

recon = holo.process(hologram)
assert np.shape(holo.window) == np.shape(recon)

# Check that autoWindow results in the correct size window being created when 
# we call update_auto_window with downsampling
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                downsample = 2,
                autoWindow = True)

holo.update_auto_window(hologram)
wind = holo.window
recon = holo.process(hologram)

assert np.shape(wind) == np.shape(recon)


# Check that we can set a manually created window
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth)

windowLowLevel = pyh.square_cosine_window(np.shape(hologram)[0], 100, 10)

holo.set_window(windowLowLevel)
recon1 = holo.process(hologram)

holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                autoWindow = True,
                windowRadius = 100)
recon2 = holo.process(hologram)


assert np.array_equal(recon1, recon2) == True



# Check that we can a window using create_window
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth)

holo.create_window(hologram, 100, 10)

recon1 = holo.process(hologram)

holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                autoWindow = True,
                windowRadius = 100)
recon2 = holo.process(hologram)


assert np.array_equal(recon1, recon2) == True



# Check that we can can apply a post refoucsing window
holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                autoWindow = True,
                postWindow = True)

recon1 = holo.process(hologram)

holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                wavelength = wavelength, 
                pixelSize = pixelSize,
                depth = depth,
                autoWindow = True)
recon2 = holo.process(hologram) 

recon2.imag = recon2.imag * holo.window
recon2.real = recon2.real * holo.window
recon2[recon2 == -0+0j] = 0j

assert np.array_equal(recon1, recon2) == True

