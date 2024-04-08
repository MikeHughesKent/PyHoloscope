# -*- coding: utf-8 -*-
"""
Tests off_axis_find_mod and off_axis_find_crop_radius

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import math
from pathlib import Path

import context                    # Relative paths

import pyholoscope as pyh

# Experimental Parameters
wavelength = 630e-9
pixelSize = .6e-6

# Load images

hologram = pyh.load_image(Path('test data/tissue_paper_oa.tif'))
background = pyh.load_image(Path('test data/tissue_paper_oa_background.tif'))

# Determine Modulation
cropCentre = pyh.off_axis_find_mod(background)
cropRadius = pyh.off_axis_find_crop_radius(background)
