# -*- coding: utf-8 -*-
"""
Example of how to use the low-level autofocus functionality of PyHoloscope.

This example attempts to refocus the whole image and scores either using a region
of interest.

For examples of faster approaches see autofocus_low_level_example.py

This example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.

Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

The focus distance estimate is then used to refcous the image and the refocused
image is displayed.

"""

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path

import context  # Loads relative paths

import pyholoscope as pyh

wavelength = 520e-9
pixel_size = 4.8e-6 
depth_range = (.08, .16)    # Will look for focus in this range only
method = 'sum'              # Can be changed to other methods

# A region of interest (ROI) around the interesting part of the image
roi = pyh.Roi(450,110,500,400)

# Load hologram and background images
holo_file = r"..\example_data\inline_paramecium\paramecium.tif"
back_file = r"..\example_data\inline_paramecium\background.tif"

hologram = pyh.load_image(holo_file)
background = pyh.load_image(back_file)


# Find focus
autofocus = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     background=background,
     roi=roi,
)
print(f"Best focus at: {round(1000 * autofocus,3)} um")


# Display hologram with ROI
fig = plt.figure(dpi=300)
plt.title('Hologram')
ax = plt.imshow(hologram, cmap = 'gray')
if roi is not None:
        plt.gca().add_patch(Rectangle((roi.x, roi.y), roi.width, roi.height, edgecolor='red',
        facecolor='none'))

# Refocus hologram using autofocus position and display
prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"Autofocused Image"); plt.imshow(refocused, cmap="gray")

