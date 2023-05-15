# -*- coding: utf-8 -*-
"""
Tests utilities functions of PyHoloscope.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import numpy as np
from pyholoscope import get8bit, get16bit, amplitude, phase, extract_central, circ_cosine_window, dimensions

# Test get8it, get16bit, amplitude, phase
img = np.ones((100,100), dtype = 'complex64')
amp = amplitude(img)
phase = phase(img)
amp, phase = get8bit(img)
amp, phase = get16bit(img)

# Test extract central
boxSize = 50
cropped = extract_central(img, boxSize)
boxSize = 100
cropped = extract_central(img, boxSize)
boxSize = 200
cropped = extract_central(img, boxSize)

# Test circ_cosine_window
imgSize = 100
circleRadius = 90
skinThickness = 10
window = circ_cosine_window(imgSize, circleRadius, skinThickness)


# Test dimensions
w = 2
h = 4
a = np.zeros((h, w))
assert dimensions(np.zeros((h, w))) == (w, h)
assert dimensions((w,h)) == (w, h)
assert dimensions((w)) == (w,w) 
