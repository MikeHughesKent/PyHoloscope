# -*- coding: utf-8 -*-
"""
PyHoloscope: prop_lut

Provides class for look-up-able for propagators used by angular spectrum 
method for numerical refocusing of holograms.

@author: Mike Hughes
Applied Optics Group, Physics & Astronomy, University of Kent
"""

import numpy as np
import pyholoscope
from pyholoscope.focusing import propagator

################## Class for LUT of propagators ##############################
class PropLUT:
    
    def __init__(self, imgSize, wavelength, pixelSize, depthRange, nDepths):
        self.depths = np.linspace(depthRange[0], depthRange[1], nDepths)
        self.size = imgSize
        self.nDepths = nDepths
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        self.propTable = np.zeros((nDepths, imgSize, imgSize), dtype = 'complex128')
        for idx, depth in enumerate(self.depths):
            self.propTable[idx,:,:] = PyHoloscope.propagator(imgSize, wavelength, pixelSize, depth)
            
    def __str__(self):
        return "LUT of " + str(self.nDepths) + " propagators from depth of " + str(self.depths[0]) + " to " + str(self.depths[-1]) + ". Wavelength: " + str(self.wavelength) + ", Pixel Size: " + str(self.pixelSize) + " ,Size:" + str(self.size)
            
    def propagator(self, depth): 
        
        # Find nearest propagator
        if depth < self.depths[0] or depth > self.depths[-1]:
            return - 1
        idx = round((depth - self.depths[0]) / (self.depths[-1] - self.depths[0]) * (self.nDepths - 1))
        #print("Desired Depth: ", depth, "Used Depth:", self.depths[idx])
        return self.propTable[idx, :,:]