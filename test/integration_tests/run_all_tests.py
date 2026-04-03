# -*- coding: utf-8 -*-
"""
Runs all integration tests of PyHoloscope.

"""

import context  # Relative paths


print("# Inline Holography")
import test_inline

print("# Inline Holography Depth Stack")
import test_inline_depth_stack

print("# Inline Holography: Angular Spectrum vs Fresnel")
import test_inline_propagation_methods

print("# Inline Holo Class: Angular Spectrum vs Fresnel")
import test_inline_holo_class_propagation_methods

print("# Off Axis Holography OOP")
import test_off_axis

print("# Off Axis Holography Low Level")
import test_off_axis_low_level

print("# Off Axis Focusing")
import test_off_axis_focusing

print("# Off Axis Focusing Low Level")
import test_off_axis_focusing_low_level

print("# Off Axis Low Level: Angular Spectrum vs Fresnel")
import test_off_axis_propagation_methods_low_level

print("# Relative phase")
import test_relative_phase

print("# Numba")
import test_numba
