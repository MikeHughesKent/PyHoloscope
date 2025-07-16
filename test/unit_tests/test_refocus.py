# -*- coding: utf-8 -*-
"""
Test refocus function.

"""

import numpy as np
import unittest
import scipy
import context

import pyholoscope as pyh


class TestRefocus(unittest.TestCase):
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
            geometry="plane",
            precision="single",
        )
        self.img_refocus = pyh.refocus(self.img, prop)

        self.assertTupleEqual(np.shape(self.img), np.shape(self.img_refocus))
        self.assertEqual(np.max(np.isnan(self.img_refocus)), 0)

    def test_plane_double_precision(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="plane",
            precision="double",
        )
        self.img_refocus = pyh.refocus(self.img, prop)

        self.assertTupleEqual(np.shape(self.img), np.shape(self.img_refocus))
        self.assertEqual(np.max(np.isnan(self.img_refocus)), 0)

    def test_point_double_precision(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="point",
            precision="double",
        )
        self.img_refocus = pyh.refocus(self.img, prop)

        self.assertTupleEqual(np.shape(self.img), np.shape(self.img_refocus))
        self.assertEqual(np.max(np.isnan(self.img_refocus)), 0)


    # Check fourier_domain = True works correctly
    def test_fourier_domain(self):
        prop = pyh.propagator(
            (self.grid_size1, self.grid_size2),
            self.wavelength,
            self.pixel_size,
            self.depth,
            geometry="plane",
            precision="single",
        )
        img_fft = scipy.fft.fft2(self.img)
        self.img_refocus1 = pyh.refocus(self.img, prop)
        self.img_refocus2 = pyh.refocus(img_fft, prop, fourier_domain = True)
        assert((self.img_refocus1==self.img_refocus2).all())

    def test_refocus_stack(self):
        
        first_depth = 0.0
        last_depth = 1.0
        depth_range = (first_depth, last_depth)
        num_depths = 10
        stack = pyh.refocus_stack(self.img, self.wavelength, self.pixel_size, 
                                  depth_range, num_depths)
        
        self.assertEqual(stack.num_depths, num_depths)
        self.assertEqual(stack.min_depth, depth_range[0])
        self.assertEqual(stack.max_depth, depth_range[1])
        self.assertEqual(np.shape(stack.stack)[0], num_depths)
        self.assertEqual(np.shape(stack.stack)[1], np.shape(self.img)[0])
        self.assertEqual(np.shape(stack.stack)[2], np.shape(self.img)[1])
        self.assertEqual(stack.stack.dtype, 'complex64')


        stack = pyh.refocus_stack(self.img, self.wavelength, self.pixel_size, 
                                    depth_range, num_depths, precision = 'double')
        self.assertEqual(stack.stack.dtype, 'complex128')
        
        stack = pyh.refocus_stack(self.img, self.wavelength, self.pixel_size, 
                                    depth_range, num_depths, geometry = 'plane')
        
        # Check first depth is correct
        prop = pyh.propagator(
          np.shape(self.img),
          self.wavelength,
          self.pixel_size,
          first_depth,
          geometry="plane",
          precision="single",)

        single_refocus = pyh.refocus(self.img, prop)
        assert((single_refocus==stack.stack[0,:,:]).all())
        
        
        # Check last depth is correct
        prop = pyh.propagator(
          np.shape(self.img),
          self.wavelength,
          self.pixel_size,
          last_depth,
          geometry="plane",
          precision="single",)

        single_refocus = pyh.refocus(self.img, prop)
 
        assert((single_refocus==stack.stack[-1,:,:]).all())
        
        # Check pre-processing is applied
        
        window = pyh.square_cosine_window(self.img, 100, 20)

        stack = pyh.refocus_stack(self.img, self.wavelength, self.pixel_size, 
                                    depth_range, num_depths, background = self.background, normalise = self.background, downsample = 2, window = window)
        
        prop = pyh.propagator(
          (np.shape(self.img)[0]/2 , np.shape(self.img)[1] /2),
          self.wavelength,
          self.pixel_size,
          last_depth,
         )
        single_refocus = pyh.refocus(self.img, prop, background = self.background, normalise = self.background, downsample = 2, window = window)
        
        
        assert((single_refocus==stack.stack[-1,:,:]).all())

 
        
        



if __name__ == "__main__":
    unittest.main()
