# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holgoraphic microscopy

PropLUT is the class for generating and storing propagator look up tables.

"""

import numpy as np

from pyholoscope.focusing import propagator
from pyholoscope.focusing_numba import propagator_numba
from pyholoscope.utils import dimensions


class PropLUT:
    """ Stores a propagator look up table (LUT). 
    
    The LUT contains angular spectrum propagators for the specified parameters. 
    """
   
    def __init__(self, imgSize, wavelength, pixelSize, depthRange, nDepths, numba = False, precision = 'single'):
        """ Create the propagator Look-Up-Table (LUT). 

        Arguments:
            imgSize   : int or tuple of (int, int)
                        size of propagators, square or rectangular
            wavelength: float
                        wavelength of light
            pixelSize : float
                        physical size of pixels
            depthRange: tuple
                        range of depths to generate propagators for
            nDepths   : int
                        number of depths to generate propagators for

        Keyword Arguments:
            numba     : bool
                        flag to use numba for speed up (default = False)
            precision : str
                        'single' or 'double' (default = 'single')
        """
        
        if precision == 'double':
            dataType = 'complex128'
        else:
            dataType = 'complex64'
        
        self.depths = np.linspace(depthRange[0], depthRange[1], nDepths)
        self.size = imgSize
        self.nDepths = nDepths
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        w,h = dimensions(imgSize)
        self.propTable = np.zeros((nDepths, h, w), dtype = dataType)
        for idx, depth in enumerate(self.depths):
            if numba is True:
                self.propTable[idx,:,:] = propagator_numba((w,h), wavelength, pixelSize, depth)
            else:
                self.propTable[idx,:,:] = propagator((w,h), wavelength, pixelSize, depth)

 
    def propagator(self, depth): 
        """ Returns the propagator from the LUT which is closest to requested 
        depth. If depth is outside the range of the propagators, function returns None.
        
        Parameters:
            depth     : float
                        refocus depth for requested propagator
        """
        
        # Find nearest propagator
        if self.nDepths == 1:   # Otherwise the algorithm to get the index will fail 
            return self.propTable[0,:,:]
        if depth < self.depths[0] or depth > self.depths[-1]:
            return None
       
        idx = round((depth - self.depths[0]) / (self.depths[-1] - self.depths[0]) * (self.nDepths - 1))

        return self.propTable[idx, :,:]
    
    
    def __str__(self):
        return "LUT of " + str(self.nDepths) + " propagators from depth of " + str(self.depths[0]) + " to " + str(self.depths[-1]) + ". Wavelength: " + str(self.wavelength) + ", Pixel Size: " + str(self.pixelSize) + " ,Size:" + str(self.size)
