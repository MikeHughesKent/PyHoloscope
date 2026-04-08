# -*- coding: utf-8 -*-
"""
Test propagator creation

"""

import unittest

import numpy as np

import context
import pyholoscope as pyh


class TestPropagator(unittest.TestCase):
    grid_size1 = 512
    grid_size2 = 1024
    wavelength = 500e-9
    pixel_size = 2e-6
    depth = 0.001

    def test_propagator(self):
        prop = pyh.propagator(
            self.grid_size1,
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="single",
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert prop.wavelength == self.wavelength
        assert prop.pixel_size == self.pixel_size
        assert prop.depth == self.depth

        assert np.shape(prop.propagator) == (self.grid_size1, self.grid_size1)
        assert prop.shape == (self.grid_size1, self.grid_size1)

        assert prop.propagator.dtype == "complex64"

        prop = pyh.propagator(
            self.grid_size1,
            self.wavelength,
            self.pixel_size,
            -self.depth,
            precision="single",
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert np.shape(prop) == (self.grid_size1, self.grid_size1)
        assert prop.propagator.dtype == "complex64"

        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="single",
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert np.shape(prop) == (self.grid_size1, self.grid_size2)
        assert prop.propagator.dtype == "complex64"

        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="double",
            use_numba=False,
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert prop.shape == (self.grid_size1, self.grid_size2)
        assert prop.propagator.dtype == "complex128"

    def test_propagator_numba(self):
        # Numba and regular are the same, double precision
        propNumba = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="double",
            use_numba=True,
        )
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="double",
        )
        assert prop.shape == (self.grid_size1, self.grid_size2)

        # Numba and regular are the same, single precision
        propNumba = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="single",
            use_numba=True,
        )
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision="single",
        )
        assert prop.shape == (self.grid_size1, self.grid_size2)

    def test_fresnel_propagator(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            propagation_method="fresnel",
            precision="single",
            use_numba=False,
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert prop.shape == (self.grid_size1, self.grid_size2)
        assert prop.propagation_method == "fresnel"

    def test_fresnel_propagator_numba(self):
        prop_numba = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            propagation_method="fresnel",
            precision="single",
            use_numba=True,
        )
        prop_numpy = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            propagation_method="fresnel",
            precision="single",
            use_numba=False,
        )

        assert np.allclose(prop_numba.propagator, prop_numpy.propagator)

    def test_correct_pixel_size(self):
        source_distance = 0.02
        effective_magnification = source_distance / (source_distance - self.depth) 
        corrected_pixel_size = self.pixel_size / effective_magnification
        corrected_depth = self.depth/effective_magnification

        prop_corrected = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            correct_pixel_size=True,
            source_distance=source_distance,
            use_numba=False,
        )

        prop_manual = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            corrected_pixel_size,
            corrected_depth,
            use_numba=False,
        )

        assert np.allclose(prop_corrected.propagator, prop_manual.propagator)
        assert np.isclose(prop_corrected.magnified_pixel_size, corrected_pixel_size)
        assert np.isclose(prop_corrected.pixel_size, self.pixel_size)
        assert np.isclose(prop_corrected.magnified_depth, corrected_depth)
        assert np.isclose(prop_corrected.depth, self.depth)
        assert prop_corrected.correct_pixel_size is True
        assert np.isclose(prop_corrected.source_distance, source_distance)

    def test_correct_pixel_size_requires_source_distance(self):
        with self.assertRaises(Exception):
            pyh.propagator(
                (self.grid_size1, self.grid_size2),
                self.wavelength,
                self.pixel_size,
                self.depth,
                correct_pixel_size=True,
                use_numba=False,
            )


if __name__ == "__main__":
    unittest.main()
