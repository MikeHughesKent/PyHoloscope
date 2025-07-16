# -*- coding: utf-8 -*-
"""
Tests off-axis holography functions of PyHoloscope

"""

import math
import unittest

import numpy as np
import matplotlib.pyplot as plt

import context

import pyholoscope as pyh
import pyholoscope.sim
from pyholoscope.utils import circ_cosine_window, circ_window


class TestOffAxis(unittest.TestCase):
    def test_predict_tilt_angle(self):
        """Generates simulated hologram then determine tilt angle
        from FFT and checks
        """

        grid_size2 = 256
        grid_size1 = 512

        angle = math.radians(3.6)

        pixel_size = 2e-6
        wavelength = 500e-9

        object_field = np.ones((grid_size2, grid_size1))
        test_hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=math.pi / 4
        )

        measured_angle = pyh.off_axis_predict_tilt_angle(
            test_hologram, wavelength, pixel_size
        )

        self.assertAlmostEqual(angle, measured_angle, places=2)

    def test_find_mod(self):
        """Generates hologram and compares detected peak in FFT with predicted
        peak position.
        """

        grid_size2 = 512
        grid_size1 = 1024

        angle = math.radians(3)
        rotation = 0.22 * math.pi
        pixel_size = 2e-6
        wavelength = 500e-9

        object_field = np.ones((grid_size2, grid_size1))
        test_hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        measured_peak_loc = pyh.off_axis_find_mod(test_hologram)
        predicted_peak_loc = pyh.off_axis_predict_mod(
            wavelength, pixel_size, (grid_size1, grid_size2), angle, rotation=rotation
        )

        self.assertAlmostEqual(predicted_peak_loc[0], measured_peak_loc[0], delta=1)
        self.assertAlmostEqual(predicted_peak_loc[1], measured_peak_loc[1], delta=1)

        measured_peak_distance = math.sqrt(
            measured_peak_loc[0] ** 2 + measured_peak_loc[1] ** 2
        )
        predicted_peak_dist = pyh.off_axis_predict_mod_distance(
            wavelength, pixel_size, (grid_size1, grid_size2), angle, rotation=rotation
        )

        self.assertAlmostEqual(measured_peak_distance, predicted_peak_dist, delta=1)

        # Check it works when modulation peak in 2nd quadrant of FFT
        rotation = 0.7 * math.pi
        test_hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        measured_peak_loc = pyh.off_axis_find_mod(test_hologram)
        predicted_peak_loc = pyh.off_axis_predict_mod(
            wavelength, pixel_size, (grid_size1, grid_size2), angle, rotation=rotation
        )

        self.assertAlmostEqual(predicted_peak_loc[0], measured_peak_loc[0], delta=1)
        self.assertAlmostEqual(predicted_peak_loc[1], measured_peak_loc[1], delta=1)

        measured_peak_distance = math.sqrt(
            measured_peak_loc[0] ** 2 + measured_peak_loc[1] ** 2
        )
        predicted_peak_dist = pyh.off_axis_predict_mod_distance(
            wavelength, pixel_size, (grid_size1, grid_size2), angle, rotation=rotation
        )

        self.assertAlmostEqual(measured_peak_distance, predicted_peak_dist, delta=1)

    def test_off_axis_demod(self):
        """Check standard demo with and without window"""

        grid_size2 = 512
        grid_size1 = 512
        pixel_size = 1e-6
        wavelength = 550e-9
        rotation = math.pi / 4
        angle = math.radians(15)

        x = 100
        y = 70
        w = 120
        h = 150

        object_field = np.zeros((grid_size2, grid_size1))
        object_field[y : y + h, x : x + w] = 1

        test_hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        crop_centre = pyh.off_axis_find_mod(test_hologram)
        crop_radius = pyh.off_axis_find_crop_radius(test_hologram)

        # Check recon has a bright square, same as object
        scale_factor = crop_radius[0] / grid_size2 * 2
        x2 = round(x * scale_factor)
        y2 = round(y * scale_factor)
        w2 = round(w * scale_factor)
        h2 = round(h * scale_factor)

        # no window
        recon = pyh.off_axis_demod(test_hologram, crop_centre, crop_radius)

        # compare mean value in the square to mean value somewhere elese (that should be zero)
        assert np.mean(
            pyh.amplitude(recon[y2 : y2 + h2, x2 : x2 + w2])
        ) > 100 * np.mean(pyh.amplitude(recon[4 * y2 : 4 * y2 + h2, x2 : x2 + w2]))

        # with window
        window = circ_cosine_window(crop_radius[0] * 2, crop_radius[0] - 10, 10)
        recon = pyh.off_axis_demod(test_hologram, crop_centre, crop_radius, mask=window)

        # compare mean value in the square to mean value somewhere elese (that should be zero)
        assert np.mean(
            pyh.amplitude(recon[y2 : y2 + h2, x2 : x2 + w2])
        ) > 100 * np.mean(pyh.amplitude(recon[4 * y2 : 4 * y2 + h2, x2 : x2 + w2]))

    def test_off_axis_demod_full(self):
        """Check demodulation and return of image same size as hologram"""

        grid_size2 = 512
        grid_size1 = 512
        pixel_size = 1e-6
        wavelength = 550e-9
        rotation = 3 * math.pi / 4
        angle = math.radians(15)

        x = 100
        y = 70
        w = 120
        h = 150

        object_field = np.zeros((grid_size2, grid_size1))
        object_field[y : y + h, x : x + w] = 1

        test_hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        crop_centre = pyh.off_axis_find_mod(test_hologram)
        crop_radius = pyh.off_axis_find_crop_radius(test_hologram)

        # no window
        recon = pyh.off_axis_demod(
            test_hologram, crop_centre, crop_radius, return_full=True
        )
        # compare mean value in the square to mean value somewhere else (that should be zero).
        assert np.mean(pyh.amplitude(recon[y : y + h, x : x + w])) > 100 * np.mean(
            pyh.amplitude(recon[4 * y : 4 * y + h, x : x + w])
        )

        # with circ window
        window = circ_window(crop_radius[0] * 2, crop_radius[0])
        recon = pyh.off_axis_demod(
            test_hologram, crop_centre, crop_radius, mask=window, return_full=True
        )
        # compare mean value in the square to mean value somewhere else (that should be very close to zero)
        assert np.mean(pyh.amplitude(recon[y : y + h, x : x + w])) > 100 * np.mean(
            pyh.amplitude(recon[4 * y : 4 * y + h, x : x + w])
        )

        # with cos window
        window = circ_cosine_window(crop_radius[0] * 2, crop_radius[0], 10)
        recon = pyh.off_axis_demod(
            test_hologram, crop_centre, crop_radius, mask=window, return_full=True
        )
        # compare mean value in the square to mean value somewhere elese (that should be zero)
        assert np.mean(pyh.amplitude(recon[y : y + h, x : x + w])) > 100 * np.mean(
            pyh.amplitude(recon[4 * y : 4 * y + h, x : x + w])
        )

    def test_off_axis_demod_rectangular(self):
        """Checks demod and return of image for non-square hologram"""

        grid_size2 = 512
        grid_size1 = 1024
        pixel_size = 1e-6
        wavelength = 550e-9
        rotation = math.pi / 4
        angle = math.radians(15)

        x = 100
        y = 70
        w = 120
        h = 150

        object_field = np.zeros((grid_size2, grid_size1))
        object_field[y : y + h, x : x + w] = 1

        test_hologram = pyh.sim.off_axis(
            object_field, wavelength, pixel_size, angle, rotation=rotation
        )

        crop_centre = pyh.off_axis_find_mod(test_hologram)
        crop_radius = pyh.off_axis_find_crop_radius(test_hologram)

        # no window
        recon = pyh.off_axis_demod(
            test_hologram, crop_centre, crop_radius, return_full=True
        )
        # compare mean value in the square to mean value somewhere elese (that should be zero)
        assert np.mean(pyh.amplitude(recon[y : y + h, x : x + w])) > 100 * np.mean(
            pyh.amplitude(recon[4 * y : 4 * y + h, x : x + w])
        )

        # with cosine window
        window = circ_cosine_window(
            (crop_radius[0] * 2, crop_radius[1] * 2), crop_radius, 10
        )
        recon = pyh.off_axis_demod(
            test_hologram, crop_centre, crop_radius, mask=window, return_full=True
        )
        # compare mean value in the square to mean value somewhere elese (that should be zero)
        assert np.mean(pyh.amplitude(recon[y : y + h, x : x + w])) > 100 * np.mean(
            pyh.amplitude(recon[4 * y : 4 * y + h, x : x + w])
        )


if __name__ == "__main__":
    unittest.main()
