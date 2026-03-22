# -*- coding: utf-8 -*-
"""
Advanced autofocus example using the Holo class.

This mirrors the low-level advanced autofocus example and demonstrates:
    1. Using full image
    2. Using a region of interest (ROI) to score, but refocusing whole image
    3. Using a ROI and only refocusing the ROI plus a margin
    4. Using full image and propagator look up table (LUT)
    5. Using a ROI to score, but refocusing whole image using LUT
    6. Using a ROI and only refocusing the ROI plus a margin using LUT
    
The example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.

Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

A Holo class object is created and used to find the focus of the hologram and return
a refocused image. Note the timings for approaches using a refocus margin
are slightly longer than the low-level example as the Holo class autofocus
functions also calculates and returns the entire refocused image after finding 
focus. 
        
"""

import time

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import context  # Loads relative paths

import pyholoscope as pyh

wavelength = 520e-9
pixel_size = 4.8e-6
depth_range = (0.08, 0.16)  # Will look for focus in this range only
num_depths = 100           # For propagator look up table
margin = 50                # For faster refocusing only around ROI + margin
method = "sum"             # Can be changed to other methods

# A region of interest (ROI) around the interesting part of the image
roi = pyh.Roi(450, 110, 500, 400)

# Load hologram and background images
holo_file = r"..\example_data\inline_paramecium\paramecium.tif"
back_file = r"..\example_data\inline_paramecium\background.tif"

hologram = pyh.load_image(holo_file)
background = pyh.load_image(back_file)

# Create Holo object
holo = pyh.Holo(
    mode=pyh.INLINE,
    wavelength=wavelength,
    pixel_size=pixel_size,
    background=background,
)

# 1. Find focus using whole image
holo.clear_propagator_LUT()
holo.set_find_focus_parameters(depth_range=depth_range, method=method)

t1 = time.perf_counter()
refocused1 = holo.auto_focus(hologram)
t2 = time.perf_counter()
autofocus1 = holo.depth
print(
    f"Full image, focus: {round(1000 * autofocus1,3)} mm, time: {round(1000 * (t2 - t1),1)} ms"
)

# 2. Find focus using scoring only the ROI
holo.clear_propagator_LUT()
holo.set_find_focus_parameters(depth_range=depth_range, method=method, roi=roi)

t1 = time.perf_counter()
refocused2 = holo.auto_focus(hologram)
t2 = time.perf_counter()
autofocus2 = holo.depth
print(
    f"ROI, focus: {round(1000 * autofocus2,3)} mm, time: {round(1000 * (t2 - t1),1)} ms"
)

# 3. Find focus using scoring only the ROI and focusing only the ROI + margin
holo.clear_propagator_LUT()
holo.set_find_focus_parameters(depth_range=depth_range, method=method, roi=roi, margin=margin)

t1 = time.perf_counter()
refocused3 = holo.auto_focus(hologram)
t2 = time.perf_counter()
autofocus3 = holo.depth
print(
    f"ROI with margin, focus: {round(1000 * autofocus3,3)} mm, time: {round(1000 * (t2 - t1),1)} ms"
)

# 4. Find focus using whole image and propagator LUT
holo.make_propagator_LUT(hologram, depth_range, num_depths)
holo.set_find_focus_parameters(depth_range=depth_range, method=method)

t1 = time.perf_counter()
refocused4 = holo.auto_focus(hologram)
t2 = time.perf_counter()
autofocus4 = holo.depth
print(
    f"Full image with LUT, focus: {round(1000 * autofocus4,3)} mm, time: {round(1000 * (t2 - t1),1)} ms"
)

# 5. Find focus scoring only the ROI and using propagator LUT
holo.make_auto_focus_propagator_LUT(hologram, depth_range, num_depths, roi=roi)
holo.set_find_focus_parameters(depth_range=depth_range, method=method, roi=roi, use_prop_lut = True)

t1 = time.perf_counter()
refocused5 = holo.auto_focus(hologram)
t2 = time.perf_counter()
autofocus5 = holo.depth
print(
    f"ROI with LUT, focus: {round(1000 * autofocus5,3)} mm, time: {round(1000 * (t2 - t1),1)} ms"
)

# 6. Find focus using scoring only the ROI and focusing only the ROI + margin using LUT
holo.make_auto_focus_propagator_LUT(hologram, depth_range, num_depths, roi=roi, margin=margin)
holo.set_find_focus_parameters(depth_range=depth_range, method=method, roi=roi, margin=margin, use_prop_lut = True)

t1 = time.perf_counter()
refocused6 = holo.auto_focus(hologram)
t2 = time.perf_counter()
autofocus6 = holo.depth
print(
    f"ROI with margin and LUT, focus: {round(1000 * autofocus6,3)} mm, time: {round(1000 * (t2 - t1),1)} ms"
)



# Display hologram with ROI
fig = plt.figure(dpi=300)
plt.title("Hologram")
ax = plt.imshow(hologram, cmap="gray")
if roi is not None:
    plt.gca().add_patch(
        Rectangle(
            (roi.x, roi.y),
            roi.width,
            roi.height,
            edgecolor="red",
            facecolor="none",
        )
    )

# Display refocused images
plt.figure(dpi=300)
plt.title("Full Image")
plt.imshow(pyh.amplitude(refocused1), cmap="gray")

plt.figure(dpi=300)
plt.title("ROI")
plt.imshow(pyh.amplitude(refocused2), cmap="gray")

plt.figure(dpi=300)
plt.title("ROI, Focus Margin")
plt.imshow(pyh.amplitude(refocused3), cmap="gray")

plt.figure(dpi=300)
plt.title("Full Image, LUT")
plt.imshow(pyh.amplitude(refocused4), cmap="gray")

plt.figure(dpi=300)
plt.title("ROI, LUT")
plt.imshow(pyh.amplitude(refocused5), cmap="gray")

plt.figure(dpi=300)
plt.title("ROI, Focus Margin, LUT")
plt.imshow(pyh.amplitude(refocused6), cmap="gray")
