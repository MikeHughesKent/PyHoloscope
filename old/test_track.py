# -*- coding: utf-8 -*-
"""
Tests tracking functionality

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

wavelength = 450e-9
pixelSize = 0.44e-6

mHolo = holo.Holo(wavelength, pixelSize)

mTrack = track.Tracker()

mTrack.test()