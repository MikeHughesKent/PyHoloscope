# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 23:14:14 2021

Tests synthetic DIC and phase gradient.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""
import numpy as np
import math
from matplotlib import pyplot as plt

import context      # Load paths

import PyHoloscope as holo



imageField = np.ones((100,100), dtype = complex)
phaseField = np.zeros((100,100), dtype = complex)

phaseField[20:40,20:40] = math.pi
imageField = imageField * np.exp(1j * phaseField)

plt.figure()
plt.imshow(np.angle(imageField))
plt.title('Phase')


#DIC
dic= holo.syntheticDIC(imageField)

plt.figure()
plt.imshow(dic)
plt.title('DIC')


# Phase Gradient
phaseGrad = holo.phaseGradient(imageField)

plt.figure()
plt.imshow(phaseGrad)
plt.title('Phase Gradient')