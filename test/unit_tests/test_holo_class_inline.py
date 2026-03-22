# -*- coding: utf-8 -*-
"""
Test that holo_class produces same output as low-level functions

"""

import numpy as np
import unittest
import scipy
import context

import pyholoscope as pyh


class TestHoloClassInline(unittest.TestCase):
    grid_size1 = 512
    grid_size2 = 1024
    wavelength = 500e-9
    pixel_size = 2e-6
    depth = 0.001

    rng = np.random.default_rng()
    img = rng.standard_normal((grid_size1, grid_size2)).astype("float32")
    background = rng.standard_normal((grid_size1, grid_size2)).astype("float32")

    def test_plane_single_precision(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
        )
        img_refocus = pyh.refocus(self.img, prop)

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            geometry="point",
        )
        img_refocus_oop = holo.process(self.img)

        assert (img_refocus == img_refocus_oop).all()

    def test_plane_double_precision(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="plane",
            precision="double",
        )
        img_refocus = pyh.refocus(self.img, prop)

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            geometry="plane",
            precision="double",
        )

        img_refocus_oop = holo.process(self.img)
        assert (img_refocus == img_refocus_oop).all()

    def test_point_double_precision(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="double",
        )
        img_refocus = pyh.refocus(self.img, prop)

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            geometry="point",
            precision="double",
        )

        img_refocus_oop = holo.process(self.img)
        assert (img_refocus == img_refocus_oop).all()

    def test_background_normalisation(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="single",
        )
        img_refocus = pyh.refocus(
            self.img, prop, background=self.background, normalise=self.background
        )

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            geometry="point",
            precision="single",
            background=self.background,
            normalise=self.background,
        )

        img_refocus_oop = holo.process(self.img)
        assert (img_refocus == img_refocus_oop).all()

    def test_window(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="single",
        )
        window = pyh.square_cosine_window(self.img, 100, 10)

        img_refocus = pyh.refocus(self.img, prop, window=window)

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            window=window,
            geometry="point",
            precision="single",
        )

        img_refocus_oop = holo.process(self.img)
        assert (img_refocus == img_refocus_oop).all()

    def test_auto_window(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="single",
        )
        window = pyh.square_cosine_window(self.img, radius=100, skin_thickness=10)

        img_refocus = pyh.refocus(self.img, prop, window=window)

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            auto_window=True,
            window_radius=100,
            window_thickness=10,
            geometry="point",
            precision="single",
        )

        img_refocus_oop = holo.process(self.img)
        assert (img_refocus == img_refocus_oop).all()

    def test_post_window(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="single",
        )
        window = pyh.square_cosine_window(self.img, 100, 10)

        img_refocus = pyh.refocus(self.img, prop, window=window)

        # Manually apply window post refocus
        img_refocus = pyh.pre_process(img_refocus, window=window)

        holo = pyh.Holo(
            mode=pyh.INLINE,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            depth=self.depth,
            auto_window=True,
            post_window=True,
            window_radius=100,
            window_thickness=10,
            geometry="point",
            precision="single",
        )

        img_refocus_oop = holo.process(self.img)
        assert (img_refocus == img_refocus_oop).all()

    def test_find_focus(self):
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


if __name__ == "__main__":
    unittest.main()
