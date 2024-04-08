# -*- coding: utf-8 -*-
"""
Tests estimate_tilt_angle

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import context                    # Relative paths
import math

import pyholoscope as holo

from pathlib import Path

# Experimental Parameters
wavelength = 630e-9
pixelSize = .6e-6

# Load image
background = holo.load_image(Path('test data/tissue_paper_oa_background.tif'))

# Estimate tilt angle
print('Estimated tilt angle:', round(holo.off_axis_predict_tilt_angle(background, wavelength, pixelSize) * 180 / math.pi,1), ' degrees')
