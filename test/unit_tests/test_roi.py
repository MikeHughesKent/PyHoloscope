# -*- coding: utf-8 -*-
"""
Tests Roi Class of PyHoloscope.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""


import unittest

import numpy as np

import context

from pyholoscope import Roi


class TestRoi(unittest.TestCase):


    def test_create(self):

        roi = Roi(40,50,200,300)
        assert roi.x == 40
        assert roi.y == 50
        assert roi.width == 200
        assert roi.height == 300


    def test_neg(self):
        # Check can't create with negative width or height
        roi = Roi(40,50,-2,-2)
        assert roi.width >= 0
        assert roi.height >= 0

    def test_constrain(self):
        # Check we can constrain roi not to be larger than an image
        roi = Roi(20,40,30,40)
        roi.constrain(25,0,60,65)
        assert roi.x >= 25
        assert roi.y >= 0
        assert roi.width <= 60
        assert roi.height <= 65

    def test_crop(self):
        # Create ROI that is larger than image, constrain to image then crop
        img = np.ones((100,100), dtype = 'complex64')        
        roi = Roi(90,20,40,40)
        roi.constrain(0,0,100,100)
        imgCrop = roi.crop(img)
        assert np.shape(imgCrop) == (40,10)
        
if __name__ == '__main__':
    unittest.main()          