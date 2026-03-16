# -*- coding: utf-8 -*-
"""
Test propagator LUT

"""

import unittest

import numpy as np

import context
import pyholoscope as pyh


class TestPropagatorLut(unittest.TestCase):
    grid_size1 = 512
    grid_size2 = 1024
    wavelength = 500e-9
    pixel_size = 2e-6
    depth1 = 0.002
    depth2 = 0.001
    depth3 = 0.0014

    def test_propagator_lut(self):
        """Test propagator LUT gives a propagator identical to one
        specifically generated.
        """

        # Make sure that the depth we want is one of the exact depths
        num_depths = 21
        depth_range = (0, 0.002)

        prop_lut = pyh.PropLUT(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            depth_range,
            num_depths,
            use_numba=True,
            precision="single",
        )

        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth1,
            precision="single",
        )
        prop_from_lut = prop_lut.propagator(self.depth1).propagator
        assert (prop_from_lut == prop.propagator).all()

        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth2,
            precision="single",
        )
        prop_from_lut = prop_lut.propagator(self.depth2).propagator

        assert (prop_from_lut == prop.propagator).all()

        # Request outside range of depths should return None
        prop_from_lut = prop_lut.propagator(depth_range[1] + 1)
        assert prop_from_lut is None
        prop_from_lut = prop_lut.propagator(depth_range[0] - 1)
        assert prop_from_lut is None

        assert prop_lut.closest_index(0) == 0
        assert prop_lut.closest_index(depth_range[0] - 1) is None
        assert prop_lut.closest_index(depth_range[0] + 1) is None

        # Request should give closest depth
        semi_spacing = (depth_range[1] - depth_range[0]) / num_depths / 2
        assert (
            prop_lut.depths[prop_lut.closest_index(self.depth3)] - self.depth3
            < semi_spacing
        )

    def test_holo_class_propagator_lut(self):
        """Test propagator lUT when using Holo class."""

        # Make sure that the depth we want is one of the exact depths
        num_depths = 21
        depth_range = (0, 0.002)
        hologram = np.random.random((self.grid_size1, self.grid_size2))

        holo = pyh.Holo(
            mode=pyh.INLINE,  # For inline holography
            wavelength=630e-9,  # Light wavelength, m
            pixel_size=1e-6,  # Hologram physical pixel size, m
            depth=0.001,  # Distance to refocus, m
            use_prop_lut=True,
        )

        # Refocus using LUT
        holo.make_propagator_LUT(
            (self.grid_size1, self.grid_size2), depth_range, num_depths
        )
        recon_lut = holo.process(hologram)

        # Refocus without LUT
        holo.use_prop_lut = False
        recon = holo.process(hologram)

        assert (recon == recon_lut).all()


if __name__ == "__main__":
    unittest.main()
