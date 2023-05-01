# -*- coding: utf-8 -*-
"""
Tests FocusStack Class of PyHoloscope.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
import context
import numpy as np
from pyholoscope import FocusStack

img = np.ones((100,100), dtype = 'complex64')

stack = FocusStack(img, (0,100), 20)

testImg = np.zeros((100,100), dtype = 'complex64')

# Add image at idx
stack.add_idx(testImg, 10)

# Add image outside allowed number
stack.add_idx(testImg, 12)

# Add image at depth
stack.add_depth(testImg, 80)

# Check depth to index
idx = stack.depth_to_index(10)
assert idx == 2

# Check index to depth
depth = stack.index_to_depth(3)
assert round(depth) == 16