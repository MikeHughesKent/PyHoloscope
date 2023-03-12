# -*- coding: utf-8 -*-
"""
Tests Roi Class of PyHoloscope.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import context
import numpy as np
from pyholoscope import Roi

img = np.ones((100,100), dtype = 'complex64')

roi = Roi(40,50,200,300)
print(f"Check ROI (40,50,200,200): {roi}")
assert roi.x == 40
assert roi.y == 50
assert roi.width == 200
assert roi.height == 300

# Check can't create with negative width or height
roi = Roi(40,50,-2,-2)
print(f"Check ROI (40,50,-2,-2): {roi}")
assert roi.width >= 0
assert roi.height >= 0

# Check we can constrain roi not to be larger than an image
roi = Roi(20,40,30,40)
print(f"Check ROI (20,40,30,40): {roi}")
roi.constrain(25,0,60,65)
print(f"Check constained to (0,0,60,65): {roi}")
print("")
assert roi.x >= 25
assert roi.y >= 0
assert roi.width <= 60
assert roi.height <= 65

# Create ROI that is larger than image, constrain to image then crop
roi = Roi(90,20,40,40)
roi.constrain(0,0,100,100)
imgCrop = roi.crop(img)
print(f"Check ROI (90,20,40,40): {roi}")
print(f"Check constained to (0,0,100,100): {roi}")
print(f"Shape of cropped image: {np.shape(imgCrop)}")
assert np.shape(imgCrop) == (40,10)