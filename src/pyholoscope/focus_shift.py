# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Mike Hughes, Applied Optics Group, University of Kent

PyHoloscope is a python library for holographic microscopy.

This file contains functions related to estimating depth from shifted images.

"""
import math
import numpy as np
try:
    import cupy as cp
    cudaAvailable = True
except:
    cudaAvailable = False
from PIL import Image
import cv2 as cv
import scipy
import time

import cv2 as cv

from pyholoscope.focus_stack import FocusStack
from pyholoscope.focusing_numba import propagator_numba
from pyholoscope.roi import Roi
from pyholoscope.utils import extract_central
import pyholoscope.general 

def determine_shift(img1, img2, **kwargs):
    """ Determines shift between two images by Normalised Cross Correlation (NCC). A sqaure template extracted
    from the centre of img2 is compared with a sqaure region extracted from the reference image img1. The size 
    of the template (templateSize) must be less than the size of the reference (refSize). The maximum
    detectable shift is the (refSize - templateSize) / 2.
    : param img1 : image as 2D numpy array
    : param img2 : image as 2D numpy array
    : upsample : factor to scale images by prior to template matching to
                 allow for sub-pixel registration.        
    """
    
    returnMax = kwargs.get('returnMax', False)
    upsample = kwargs.get('upsample', 1)
    
    refSize = np.shape(img1)[0] / 2
    templateSize = np.shape(img1)[0] / 4
    

    if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
        return -1
    else:

        template = extract_central(img2, templateSize).astype('float32')
        refIm = extract_central(img1, refSize).astype('float32')

        if upsample != 1:

            template = cv.resize(template, (np.shape(template)[
                                 0] * upsample, np.shape(template)[1] * upsample))
            refIm = cv.resize(
                refIm, (np.shape(refIm)[0] * upsample, np.shape(refIm)[0] * upsample))
        res = cv.matchTemplate(template, refIm, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        shift = [(max_loc[0] - (refSize - templateSize) * upsample)/upsample,
                 (max_loc[1] - (refSize - templateSize) * upsample)/upsample]
        
        if returnMax:
            return shift, max_val
        else:
            return shift
    
    
def calibrate_depth_from_shift(img1, img2, depth):
    pass

    
    