# -*- coding: utf-8 -*-
"""
Tests focus score functions.
"""

import unittest

import numpy as np

import context   # Paths
import pyholoscope as pyh


class TestFocusScores(unittest.TestCase):
    
    def setUp(self):
        self.img = np.arange(25, dtype=float).reshape((5, 5))

    def test_builtin_scores(self):
        for name in pyh.get_focus_score_methods():
            score = pyh.focus_score(self.img, name)
            assert np.isfinite(score)

    def test_callable_score(self):
        score = pyh.focus_score(self.img, lambda i: -np.mean(i))
        assert np.isfinite(score)
   
    def test_invalid_method(self):
        with self.assertRaises(Exception):
            pyh.focus_score(self.img, "DUMMY")


if __name__ == "__main__":
    unittest.main()
