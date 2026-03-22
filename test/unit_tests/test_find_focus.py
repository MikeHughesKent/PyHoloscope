# -*- coding: utf-8 -*-
"""
Tests find_focus and related functions.
"""

import unittest

import numpy as np

import context   # Paths
import pyholoscope as pyh
from pyholoscope.roi import Roi


class TestFindFocus(unittest.TestCase):
    def setUp(self):
        self.img = np.arange(15 * 12, dtype=float).reshape((15, 12))
        self.wavelength = 500e-9
        self.pixel_size = 2e-6
        self.depth_range = (-1e-3, 1e-3)

    def test_find_focus_returns_in_range(self):
        depth = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            method="Peak",
        )
        assert np.isfinite(depth)
        assert self.depth_range[0] <= depth <= self.depth_range[1]

    def test_find_focus_callable_method(self):
        depth = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            method=lambda i: -np.mean(i),
        )
        assert np.isfinite(depth)
        assert self.depth_range[0] <= depth <= self.depth_range[1]

    def test_find_focus_with_coarse_search(self):
        depth = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            method="Sum",
            coarse_search_interval=3,
        )
        assert np.isfinite(depth)
        assert self.depth_range[0] <= depth <= self.depth_range[1]

    def test_find_focus_with_roi_and_margin(self):
        roi = Roi(3, 3, 4, 4)
        depth = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            method="Sobel",
            roi=roi,
            margin=1,
        )
        assert np.isfinite(depth)
        assert self.depth_range[0] <= depth <= self.depth_range[1]
        
        

    def test_find_focus_with_prop_lut(self):
        lut = pyh.PropLUT(
            self.img.shape,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            num_depths=5,
            use_numba=False,
        )
        depth = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            method="sum",
            prop_lut=lut,
        )
        assert np.isfinite(depth)
        assert self.depth_range[0] <= depth <= self.depth_range[1]

    def test_propagator_size_for_roi(self):
        roi = Roi(1, 1, 3, 3)
        size = pyh.propagator_size_for_roi(self.img.shape, roi, margin=1)
        assert size == (5, 5)


        # Check that it works when margin exceed size of hologram
        roi = Roi(2, 2, 2, 2)
        size = pyh.propagator_size_for_roi(self.img.shape, roi, margin=40)
        assert size == self.img.shape

        size = pyh.propagator_size_for_roi(self.img.shape, None, margin=1)
        assert size == self.img.shape

        size = pyh.propagator_size_for_roi(self.img.shape, roi, margin=None)
        assert size == self.img.shape
        
        
    
    def test_find_focus_with_prop_lut_margin(self):
        
        roi = Roi(3, 3, 4, 4)
        margin = 2

        size = pyh.propagator_size_for_roi(self.img.shape, roi, margin=margin)
        
        lut = pyh.PropLUT(
            size,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            num_depths=5,
            use_numba=False,
        )
        depth = pyh.find_focus(
            self.img,
            self.wavelength,
            self.pixel_size,
            self.depth_range,
            roi = roi,
            margin = margin,
            method="sum",
            prop_lut=lut,
        )
        assert np.isfinite(depth)
        assert self.depth_range[0] <= depth <= self.depth_range[1]


if __name__ == "__main__":
    unittest.main()
