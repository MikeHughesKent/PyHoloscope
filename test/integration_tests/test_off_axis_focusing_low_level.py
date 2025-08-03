# -*- coding: utf-8 -*-
"""
Tests object oriented off axis holography with numerical refocusing
functionality of PyHoloscope using direct calling of lower-level functions.

"""

import context

from matplotlib import pyplot as plt

from pathlib import Path

import numpy as np

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixel_size = 1e-6
grid_size = 1024
depth = 0.00117

# Load images. 
hologram = pyh.load_image(Path("test data/paramecium_oa_oof.tif"))
background = pyh.load_image(Path("test data/paramecium_oa_oof_background.tif"))

# Determine Modulation
crop_centre = pyh.off_axis_find_mod(background)
crop_radius = pyh.off_axis_find_crop_radius(background)

# Remove modulation
recon_field = pyh.off_axis_demod(hologram, crop_centre, crop_radius)
background_field = pyh.off_axis_demod(background, crop_centre, crop_radius)

# Apply background correction
corrected_field = pyh.relative_phase(recon_field, background_field)

adjusted_pixel_size = pixel_size / (crop_radius[0] * 2) * np.shape(hologram)[1]

prop = pyh.propagator(
    (crop_radius[1] * 2, crop_radius[0] * 2), wavelength, adjusted_pixel_size, depth
)
refocused_field = pyh.refocus(corrected_field, prop)

# These lines can be uncommented to dump a depth stack to test.tif
# refocus_field = mHolo.refocus(hologram)
# depth_stack = mHolo.depth_stack(recon_field, (-0.0001,0.0001), 100)
# depth_stack.write_intensity_to_tif('test.tif')

# Unwrap phase
phase_unwrapped = pyh.phase_unwrap(pyh.phase(refocused_field))
tilt = pyh.obtain_tilt(phase_unwrapped)

# Remove image phase tilt
phase_untilted = phase_unwrapped - tilt

# Display intensity and phase
plt.figure(dpi=150)
plt.imshow(pyh.amp(recon_field), cmap="gray", interpolation="None")
plt.title("Amplitude (out of focus)")

plt.figure(dpi=150)
plt.imshow(pyh.phase(recon_field), cmap="twilight", interpolation="None")
plt.title("Phase")

plt.figure(dpi=150)
plt.imshow(pyh.amp(refocused_field), cmap="gray", interpolation="None")
plt.title("Refocused Amplitude")

plt.figure(dpi=150)
plt.imshow(pyh.phase(refocused_field), cmap="twilight", interpolation="None")
plt.title("Refocused Phase (Wrapped)")

plt.figure(dpi=150)
plt.imshow(phase_unwrapped, interpolation="None")
plt.title("Refocused Phase Unwrapped")

plt.figure(dpi=150)
plt.imshow(phase_untilted, interpolation="None")
plt.title("Refocused Phase Unwrapped, Tilt Removed")

DIC = pyh.synthetic_DIC(refocused_field, shear_angle=0)
plt.figure(dpi=150)
plt.imshow(DIC, cmap="gray", interpolation="None")
plt.title("Synthetic DIC")

phase_grad = pyh.phase_gradient(refocused_field)
plt.figure(dpi=150)
plt.imshow(phase_grad, cmap="gray")
plt.title("Phase Gradient")
