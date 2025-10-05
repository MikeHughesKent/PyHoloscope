# -*- coding: utf-8 -*-
"""
Tests Roi Class of PyHoloscope.

"""

import unittest

import numpy as np

import context

from pyholoscope import FocusStack


class TestFocusStack(unittest.TestCase):
    def test_focus_stack(self):
        img = np.ones((100, 100), dtype="complex64")

        stack = FocusStack(img, (0, 100), 20)

        testImg = np.zeros((100, 100), dtype="complex64")

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


if __name__ == "__main__":
    unittest.main()
