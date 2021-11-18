# -*- coding: utf-8 -*-
"""
Tests sim functionality

@author: Mike Hughes
"""

from matplotlib import pyplot as plt
import numpy as np
import time

import cv2 as cv

import context
from pybundle import PyBundle

import PyHoloscope as holo
import PyHoloscope.sim as sim

#mHolo = holo.Holo()

wavelength = 450e-9
pixelSize = 0.44e-6
tiltAngle = .2

objectField = np.ones((100,100), dtype = complex)

phaseField = np.zeros((100,100))
phaseField[40:60,30:80] = 3
objectField = objectField * np.exp(1j * phaseField)

cameraImage = sim.offAxis(objectField, wavelength, pixelSize, tiltAngle)



cropCentre = holo.offAxisPredictMod(wavelength, pixelSize, tiltAngle)
    
    
cropCentre = 65
cropRadius = 20
reconField = holo.offAxis(cameraImage, cropCentre, cropRadius)


plt.figure()
plt.imshow(cameraImage)
plt.imshow(np.angle(reconField))
