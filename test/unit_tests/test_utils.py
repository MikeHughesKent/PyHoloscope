# -*- coding: utf-8 -*-
"""
Tests utilities functions of PyHoloscope.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import unittest 

import numpy as np

from pyholoscope import get8bit, get16bit, amplitude, phase, extract_central, circ_cosine_window, dimensions


class TestUtils(unittest.TestCase):

    def test_gets(self):
        # Test get8it, get16bit, amplitude, phase
        img = np.ones((100,100), dtype = 'complex64')
        amp1 = amplitude(img)
        self.assertEqual(amp1.dtype, 'float32')
    
        phase1 = phase(img)
        self.assertEqual(phase1.dtype, 'float32')
    
        amp8, phase8 = get8bit(img)
        self.assertEqual(amp8.dtype, 'uint8')
        self.assertEqual(phase8.dtype, 'uint8')

        amp16, phase16 = get16bit(img)
        self.assertEqual(amp16.dtype, 'uint16')
        self.assertEqual(phase16.dtype, 'uint16')
        

    def test_extract_central(self):
        
        img = np.ones((100,100), dtype = 'complex64')

        boxSize = 50
        cropped = extract_central(img, boxSize)
        self.assertTupleEqual(np.shape(cropped), (boxSize * 2, boxSize * 2))
        
        boxSize = 100
        cropped = extract_central(img, boxSize)
        self.assertTupleEqual(np.shape(cropped), (100,100))
        
        boxSize = 200
        cropped = extract_central(img, boxSize)
        self.assertTupleEqual(np.shape(cropped), (100,100))


    def test_circ_cosine_window(self):
        
        imgSize = 100
        circleRadius = 40
        skinThickness = 10
        window = circ_cosine_window(imgSize, circleRadius, skinThickness)
        self.assertEqual(window[0,0], 0)
        self.assertEqual(window[50,50], 1)
        self.assertTupleEqual(np.shape(window), (imgSize, imgSize))
        
        

    def test_dimensions(self):

        w = 2
        h = 4
        a = np.zeros((h, w))
        assert dimensions(np.zeros((h, w))) == (w, h)
        assert dimensions((w,h)) == (w, h)
        assert dimensions((w)) == (w,w) 
        

if __name__ == '__main__':
    import context
    unittest.main()