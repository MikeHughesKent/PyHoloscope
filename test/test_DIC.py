# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 23:14:14 2021

@author: AOG
"""
import numpy as np
import math
from matplotlib import pyplot as plt

import PyHoloscope as holo

imageField = np.ones((100,100)) + 1j * np.ones((100,100))

xM, yM = np.meshgrid(np.linspace(0, 3 * math.pi, num = 100), np.linspace(0,0,num = 100)) 

imageField= imageField * np.exp(1j * xM)

plt.figure()
plt.imshow(np.angle(imageField))
plt.title('Phase')

DIC = holo.syntheticDIC(imageField)

plt.figure()
plt.imshow(DIC)
plt.title('DIC')