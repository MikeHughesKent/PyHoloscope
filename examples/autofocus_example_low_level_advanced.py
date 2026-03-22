# -*- coding: utf-8 -*-
"""
Examples of how to use the low-level autofocus functionality of PyHoloscope.

For simpler examples see autofocus_example.py and autofocus_low_level_example.py

This example shows six different ways to refocus:
    
    1. Using full image
    2. Using a region of interest (ROI) to score, but refocusing whole image
    3. Using a ROI and only refocusing the ROI plus a margin
    4. Using full image and propagator look up table (LUT)
    5. Using a ROI to score, but refocusing whole image using LUT
    6. Using a ROI and only refocusing the ROI plus a margin using LUT (fastest method)

The example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.

Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

The focus distance estimate is then used to refcous the image and these
are displayed for each method.

"""

import time

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path

import context  # Loads relative paths

import pyholoscope as pyh

wavelength = 520e-9
pixel_size = 4.8e-6 
depth_range = (.08, .16)    # Will look for focus in this range only
num_depths = 100            # For propagator look up table
margin = 50                 # For faster refocusing only around ROI + margin
method = 'sum'              # Can be changed to other methods

# A region of interest (ROI) around the interesting part of the image
roi = pyh.Roi(450,110,500,400)

# Load hologram and background images
holo_file = r"..\example_data\inline_paramecium\paramecium.tif"
back_file = r"..\example_data\inline_paramecium\background.tif"

hologram = pyh.load_image(holo_file)
background = pyh.load_image(back_file)


# Find focus using whole image
t1 = time.perf_counter()
autofocus1 = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     background=background,
)
t2 = time.perf_counter()
print(f"Full image, focus: {round(1000 * autofocus1,3)} mm, time: {round(1000 * (t2 -t1),1)} ms")


# Find focus using scoring only the ROI 
t1 = time.perf_counter()
autofocus2 = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     roi = roi,
     background=background,
)
t2 = time.perf_counter()
print(f"ROI, focus: {round(1000 * autofocus2,3)} mm, time: {round(1000 * (t2 -t1),1)} ms")


# Find focus using scoring only the ROI and focusing only the ROI
t1 = time.perf_counter()
autofocus3 = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     roi = roi,
     margin = margin,
     background=background,
)
t2 = time.perf_counter()
print(f"ROI with margin, focus: {round(1000 * autofocus3,3)} mm, time: {round(1000 * (t2 -t1),1)} ms")


# Find focusing using whole image and propagator LUT
prop_lut = pyh.PropLUT(hologram, wavelength, pixel_size, depth_range, num_depths)

t1 = time.perf_counter()
autofocus4 = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     roi = roi,
     background=background,
     prop_lut = prop_lut,
)
t2 = time.perf_counter()
print(f"Full image with LUT, focus: {round(1000 * autofocus4,3)} mm, time: {round(1000 * (t2 -t1),1)} ms")


# Find focus scoring only the ROI and using propagator LUT
prop_size = pyh.propagator_size_for_roi(hologram, roi = roi)
prop_lut = pyh.PropLUT(prop_size, wavelength, pixel_size, depth_range, num_depths)

t1 = time.perf_counter()
autofocus5 = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     roi = roi,
     background=background,
     prop_lut = prop_lut,
)
t2 = time.perf_counter()
print(f"ROI with margin and LUT, focus: {round(1000 * autofocus5,3)} mm, time: {round(1000 * (t2 -t1),1)} ms")




# Find focus using scoring only the ROI and focusing only the ROI using propagator LUT
prop_size = pyh.propagator_size_for_roi(hologram, roi = roi, margin = margin)
prop_lut = pyh.PropLUT(prop_size, wavelength, pixel_size, depth_range, num_depths)

t1 = time.perf_counter()
autofocus6 = pyh.find_focus(
     hologram,
     wavelength,
     pixel_size,
     depth_range,
     method,
     roi = roi,
     margin = margin,
     background=background,
     prop_lut = prop_lut,
)
t2 = time.perf_counter()
print(f"ROI with margin and LUT, focus: {round(1000 * autofocus6,3)} mm, time: {round(1000 * (t2 -t1),1)} ms")



# Display hologram with ROI
fig = plt.figure(dpi=300)
plt.title('Hologram')
ax = plt.imshow(hologram, cmap = 'gray')
if roi is not None:
        plt.gca().add_patch(Rectangle((roi.x, roi.y), roi.width, roi.height, edgecolor='red',
        facecolor='none'))

# Refocus hologram using focus positions from each method and display

prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus1)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"Full Image"); plt.imshow(refocused, cmap="gray")


prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus2)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"ROI"); plt.imshow(refocused, cmap="gray")


prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus3)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"ROI, Focus Margin"); plt.imshow(refocused, cmap="gray")


prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus4)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"Full Image, LUT"); plt.imshow(refocused, cmap="gray")


prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus6)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"ROI, LUT"); plt.imshow(refocused, cmap="gray")


prop = pyh.propagator(hologram, wavelength, pixel_size, autofocus5)
refocused = pyh.amplitude(pyh.refocus(hologram, prop, background=background))
plt.figure(dpi=300);plt.title(f"ROI, Focus Margin, LUT"); plt.imshow(refocused, cmap="gray")

