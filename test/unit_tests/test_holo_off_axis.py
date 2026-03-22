# -*- coding: utf-8 -*-
"""
Test that Holo class reproduces low-level off-axis output.

"""

import math
import unittest

import numpy as np

import context
import pyholoscope as pyh
import pyholoscope.sim


class TestHoloOffAxis(unittest.TestCase):
    def test_holo_off_axis_matches_low_level(self):
        grid_size = 512
        pixel_size = 1e-6
        wavelength = 550e-9
        rotation = math.pi / 4
        angle = math.radians(15)

        x = 100
        y = 70
        w = 120
        h = 150

        object_field = np.zeros((grid_size, grid_size))
        object_field[y : y + h, x : x + w] = 1

        hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        crop_centre = pyh.off_axis_find_mod(hologram)
        crop_radius = pyh.off_axis_find_crop_radius(hologram)

        demod_low = pyh.off_axis_demod(
            hologram,
            crop_centre,
            crop_radius,
            mask=None,
            cuda=False,
            real_fft=False,
        )

        holo = pyh.Holo(
            mode=pyh.OFF_AXIS,
            wavelength=wavelength,
            pixel_size=pixel_size,
            crop_centre=crop_centre,
            crop_radius=crop_radius,
            cuda=False,
        )

        demod_holo = holo.process(hologram)

        assert np.allclose(demod_low, demod_holo)

    def test_holo_off_axis_refocus_matches_low_level(self):
        grid_size = 512
        pixel_size = 1e-6
        wavelength = 550e-9
        rotation = math.pi / 4
        angle = math.radians(15)
        depth = 0.001

        x = 100
        y = 70
        w = 120
        h = 150

        object_field = np.zeros((grid_size, grid_size))
        object_field[y : y + h, x : x + w] = 1

        hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        crop_centre = pyh.off_axis_find_mod(hologram)
        crop_radius = pyh.off_axis_find_crop_radius(hologram)

        demod_low = pyh.off_axis_demod(
            hologram,
            crop_centre,
            crop_radius,
            mask=None,
            cuda=False,
            real_fft=False,
        )

        oa_pixel_size = pyh.off_axis_demod_pixel_size(object_field, pixel_size, crop_radius)
        prop = pyh.propagator(demod_low, wavelength, oa_pixel_size, depth)
        refocused_low = pyh.refocus(demod_low, prop)

        holo = pyh.Holo(
            mode=pyh.OFF_AXIS,
            wavelength=wavelength,
            pixel_size=pixel_size,
            crop_centre=crop_centre,
            crop_radius=crop_radius,
            refocus=True,
            depth=depth,
            cuda=False,
        )
        refocused_holo = holo.process(hologram)

        assert np.allclose(refocused_low, refocused_holo)


if __name__ == "__main__":
    unittest.main()
