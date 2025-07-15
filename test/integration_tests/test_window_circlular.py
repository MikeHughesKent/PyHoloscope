# -*- coding: utf-8 -*-
"""
Tests circular windowing functionality of PyHoloscope

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
                windowShape = 'circle',
                autoWindow = True)

holo.update_auto_window(hologram)

plt.imshow(holo.window)

