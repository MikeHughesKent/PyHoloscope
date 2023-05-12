# -*- coding: utf-8 -*-
"""
Runs all tests of PyHoloscope. 

@author: Mike Hughes. Applied Optics Group, University of Kent
"""

import context                    # Relative paths


print("# Inline Holography")
import test_inline

print("# Inline Holography Depth Stack")
import test_inline_depth_stack

print("# Off Axis Holography OOP")
import test_off_axis

print("# Off Axis Holography Low Level")
import test_off_axis_low_level

print("# Off Axis Focusing")
import test_off_axis_focusing

print("# Off Axis Focusing Low Level")
import test_off_axis_focusing_low_level

print('# Relative phase')
import test_relative_phase

print('# Relative phase refocus')
import test_relative_phase_refocus

print("# Estimate Tilt Angle")
import test_estimate_tilt_angle

print("# Find Mod Frequency and Radius")
import test_find_mod_and_radius

print("# Windowing")
import test_window

print("# Numba")
import test_numba

print("# Focus Stack")
import test_focus_stack

print("# ROI")
import test_roi

print("# Utilities")
import test_utils