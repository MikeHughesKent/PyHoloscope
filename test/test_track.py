# -*- coding: utf-8 -*-
"""
Tests numerical refocus of inline hologram.

@author: Mike Hughes
"""

from matplotlib import pyplot as plt
import numpy as np
import time

import cv2 as cv

import context
import PyHoloscope as holo
from pybundle import PyBundle

import PyHoloscope.track as track

import PyHoloscope as holo

mHolo = holo.Holo()

mTrack = track.Tracker()

mTrack.test()