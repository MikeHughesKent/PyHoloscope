# -*- coding: utf-8 -*-
"""
Test refocus function.

"""


import numpy as np
import unittest

import context

import pyholoscope as pyh


class TestPropagator(unittest.TestCase):

    gridSize1 = 512
    gridSize2 = 1024
    wavelength = 500e-9
    pixelSize = 2e-6
    depth = 0.001

    rng = np.random.default_rng()
    img = rng.standard_normal((gridSize2, gridSize1)).astype('float32')
    
    def test_plane_single_precision(self):    

        prop = pyh.propagator((self.gridSize1, self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'plane', precision = 'single')
        self.img_refocus = pyh.refocus(self.img, prop)
        
        self.assertTupleEqual( np.shape(self.img), np.shape(self.img_refocus))
        self.assertEqual(np.max(np.isnan(self.img_refocus)), 0)
        
        
    def test_plane_double_precision(self):       

        prop = pyh.propagator((self.gridSize1, self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'plane', precision = 'double')
        self.img_refocus = pyh.refocus(self.img, prop)
        
        self.assertTupleEqual( np.shape(self.img), np.shape(self.img_refocus))
        self.assertEqual(np.max(np.isnan(self.img_refocus)), 0)


    def test_point_double_precision(self):
        
        prop = pyh.propagator((self.gridSize1, self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'double')
        self.img_refocus = pyh.refocus(self.img, prop)

        self.assertTupleEqual( np.shape(self.img), np.shape(self.img_refocus))
        self.assertEqual(np.max(np.isnan(self.img_refocus)), 0)
        

if __name__ == '__main__':
    unittest.main()        