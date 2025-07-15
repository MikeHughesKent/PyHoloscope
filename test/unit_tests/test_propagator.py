# -*- coding: utf-8 -*-
"""
Test propagator creation

"""

import unittest

import numpy as np

import context
import pyholoscope as pyh


class TestPropagator(unittest.TestCase):


    gridSize1 = 512
    gridSize2 = 1024
    wavelength = 500e-9
    pixelSize = 2e-6
    depth = 0.001

    def test_propagator(self):
        
        prop = pyh.propagator(self.gridSize1, self.wavelength, self.pixelSize, self.depth, geometry = 'plane', precision = 'single')
        assert np.max(np.isnan(prop) == 0)
        assert np.shape(prop) == (self.gridSize1, self.gridSize1)
        assert(prop.dtype == 'complex64')

        prop = pyh.propagator(self.gridSize1, self.wavelength, self.pixelSize, -self.depth, geometry = 'plane', precision = 'single')
        assert np.max(np.isnan(prop) == 0)
        assert np.shape(prop) == (self.gridSize1, self.gridSize1)
        assert(prop.dtype == 'complex64')

        prop = pyh.propagator((self.gridSize1,self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'plane', precision = 'single')
        assert np.max(np.isnan(prop) == 0)
        assert np.shape(prop) == (self.gridSize2, self.gridSize1)
        assert(prop.dtype == 'complex64')

        prop = pyh.propagator((self.gridSize1, self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'single')
        assert np.max(np.isnan(prop) == 0)
        assert np.shape(prop) == (self.gridSize2, self.gridSize1)
        assert(prop.dtype == 'complex64')       
       
        prop = pyh.propagator((self.gridSize1,self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'double')
        assert np.max(np.isnan(prop) == 0)
        assert np.shape(prop) == (self.gridSize2, self.gridSize1)
        assert(prop.dtype == 'complex128')


    def test_propagator_numba(self):

        # Numba and regular are the same, point and double precision
        propNumba = pyh.propagator_numba((self.gridSize1, self.gridSize2),  self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'double')
        prop = pyh.propagator((self.gridSize1,self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'double')
        assert np.shape(prop) == (self.gridSize2, self.gridSize1)
        
        
        # Numba and regular are the same, point and single precision
        propNumba = pyh.propagator_numba((self.gridSize1, self.gridSize2),  self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'single')
        prop = pyh.propagator((self.gridSize1,self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'point', precision = 'single')
        assert np.shape(prop) == (self.gridSize2, self.gridSize1)
        
        
        # Numba and regular are the same, plane and single precision
        propNumba = pyh.propagator_numba((self.gridSize1, self.gridSize2),  self.wavelength, self.pixelSize, self.depth, geometry = 'plane', precision = 'single')
        prop = pyh.propagator((self.gridSize1,self.gridSize2), self.wavelength, self.pixelSize, self.depth, geometry = 'plane', precision = 'single')
        assert np.shape(prop) == (self.gridSize2, self.gridSize1)



if __name__ == '__main__':
    unittest.main()