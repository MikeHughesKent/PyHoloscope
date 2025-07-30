# -*- coding: utf-8 -*-
"""
Tests general internal functions of PyHoloscope.

"""

import unittest

import numpy as np
import scipy as sp

from pyholoscope import fast_full_2D_fft
import matplotlib.pyplot as plt


class TestUtils(unittest.TestCase):
    
    def test_fast_full_2D_fft(self):
        
        img = np.random.randint(0, 100, size= (40, 60)).astype('float32')
        
        fft = sp.fft.fft2(img)
       
        fft_fast = fast_full_2D_fft(img)
       
        assert np.allclose(fft, fft_fast)
        
       
        img = np.random.randint(0, 100, size=(30, 29)) 
        
        fft = sp.fft.fft2(img)
       
        fft_fast = fast_full_2D_fft(img)
       
        assert np.allclose(fft, fft_fast)
       
 


if __name__ == "__main__":
    import context

    unittest.main()
