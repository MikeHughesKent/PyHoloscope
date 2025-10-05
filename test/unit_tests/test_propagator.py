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
            geometry="plane",
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
            geometry="plane",
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
            geometry="plane",
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
            geometry="point",
            precision="single",
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert prop.shape == (self.grid_size1, self.grid_size2)
        assert prop.propagator.dtype == "complex64"

        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="double",
            use_numba=False,
        )
        assert np.max(np.isnan(prop.propagator) == 0)
        assert prop.shape == (self.grid_size1, self.grid_size2)
        assert prop.propagator.dtype == "complex128"

    def test_propagator_numba(self):
        # Numba and regular are the same, point and double precision
        propNumba = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="double",
            use_numba=True,
        )
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="double",
        )
        assert prop.shape == (self.grid_size1, self.grid_size2)

        # Numba and regular are the same, point and single precision
        propNumba = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="single",
            use_numba=True,
        )
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="single",
        )
        assert prop.shape == (self.grid_size1, self.grid_size2)

        # Numba and regular are the same, plane and single precision
        propNumba = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="plane",
            precision="single",
            use_numba=True,
        )
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="plane",
            precision="single",
        )
        assert prop.shape == (self.grid_size1, self.grid_size2)


if __name__ == "__main__":
    unittest.main()
