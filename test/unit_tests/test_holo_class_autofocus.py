# -*- coding: utf-8 -*-
"""
Tests Holo class autofocus against low-level functions.

"""

import numpy as np
import unittest

import context
import pyholoscope as pyh


class TestHoloClassAutofocus(unittest.TestCase):
    grid_size1 = 512
    grid_size2 = 1024
    wavelength = 500e-9
    pixel_size = 2e-6

    rng = np.random.default_rng()
    img = rng.standard_normal((grid_size1, grid_size2)).astype("float32")
    background = rng.standard_normal((grid_size1, grid_size2)).astype("float32")

    def test_find_focus_matches_low_level(self):
        depth_range = (0.0005, 0.0015)
        method = "sum"
        roi = pyh.Roi(100, 100, 200, 200)
        margin = 20

        depth_low = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            depth_range,
            method,
            background=self.background,
            roi=roi,
            margin=margin,
        )

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            background=self.background,
        )
        holo.set_find_focus_parameters(
            depth_range=depth_range,
            method=method,
            roi=roi,
            margin=margin,
        )
        depth_holo = holo.find_focus(self.img)

        assert np.isclose(depth_low, depth_holo)

    def test_find_focus_matches_low_level_with_prop_lut(self):
        depth_range = (0.0005, 0.0015)
        method = "sum"
        roi = pyh.Roi(100, 100, 200, 200)
        margin = 20
        num_depths = 10

        # Low-level with LUT
        prop_size = pyh.propagator_size_for_roi(self.img, roi=roi, margin=margin)
        prop_lut = pyh.PropLUT(
            prop_size,
            self.wavelength,
            self.pixel_size,
            depth_range,
            num_depths,
            use_numba=False,
        )
        depth_low = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            depth_range,
            method,
            background=self.background,
            roi=roi,
            margin=margin,
            prop_lut=prop_lut,
        )

        # Holo class with auto-focus LUT
        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            background=self.background,
        )
        holo.make_auto_focus_propagator_LUT(
            self.img,
            depth_range,
            num_depths,
            roi=roi,
            margin=margin,
        )
        holo.set_find_focus_parameters(
            depth_range=depth_range,
            method=method,
            roi=roi,
            margin=margin,
            use_prop_lut=True,
        )
        depth_holo = holo.find_focus(self.img)

        assert np.isclose(depth_low, depth_holo)


if __name__ == "__main__":
    unittest.main()
