## -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holographic microscopy

The Holo Class provides an object-oriented interface to a subset of the 
PyHoloscope functionality.

@author: Mike Hughes
Applied Optics Group, Physics & Astrfonomy, University of Kent
"""

import numpy as np
import cv2 as cv
import time

from pyholoscope.utils import circ_cosine_window, square_cosine_window
from pyholoscope.off_axis import off_axis_find_mod, off_axis_find_crop_radius, off_axis_demod
from pyholoscope.focusing import propagator, refocus, find_focus, refocus_stack
from pyholoscope.general import pre_process, post_process, relative_phase
from pyholoscope.prop_lut import PropLUT

try:
    import cupy
    cudaAvailable = True
except:
    cudaAvailable = False
    
    
try:
    import numba
    from pyholoscope.focusing_numba import propagator_numba
    numbaAvailable = True
except:
    numbaAvailable = False    

class Holo:
        
    INLINE_MODE = 1
    OFFAXIS_MODE = 2
    
    def __init__(self, mode, wavelength, pixelSize, **kwargs):
        
        self.mode = mode
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        self.oaPixelSize = pixelSize
        
        self.useNumba = kwargs.get('numba', True)
        self.cuda = kwargs.get('cuda', True)
        
        self.depth = kwargs.get('depth', 0)
        self.background = kwargs.get('background',None)
        self.applyWindow = False
        self.window = kwargs.get('window', None)        
        self.windowRadius = kwargs.get('windowRadius', None)
        self.windowThickness = kwargs.get('windowThickness', None)
        
        self.findFocusMethod = kwargs.get('findFocusMethod', 'Brenner')
        self.findFocusRoi = kwargs.get('findFocusRoi', None)
        self.findFocusMargin = kwargs.get('findFocusMargin', None)
        self.findFocusCoarseSearchInterval = kwargs.get('findFocusCoarseSearchInterval', None)
        self.findFocusDepthRange = kwargs.get('findFocusDepthRange', (0,1))
       
        self.backgroundField = None
        self.propagatorDepth = 0
        self.propagatorWavelength = 0
        self.propagatorPixelSize = 0
        self.propagatorSize = 0
        self.propagator = None
        self.propagatorLUT = None
        
        self.cropCentre = None
        self.cropRadius = None
        self.returnFFT = False
        
        self.relativePhase = False
        self.stablePhase = False
        
        # Off-axis
        self.cropCentre = (0,0)
        self.cropRadius = 0
        
        self.invert = False
        self.refocus = False
        self.downsample = 1
        self.stableROI = None
        
    def __str__(self):
        return "PyHoloscope Holo Class. Wavelength: " + str(self.wavelength) + ", Pixel Size: " + str(self.pixelSize)


    def set_depth(self, depth):
        """ Set the depth for numerical refocusing """
        self.depth = depth

        
    def set_wavelength(self, wavelength):
        """ Set the wavelength of the hologram """
        self.wavelength = wavelength

        
    def set_pixel_size(self, pixelSize):
        """ Set the size of pixels in the raw hologram"""
        self.pixelSize = pixelSize     
        
        
    def set_background(self, background):
        """ Set the background image. Use None to remove background. """
        if background is not None:
            self.background  = background.astype('uint16') 
        else:
            self.background = None
            
        
    def clear_background(self):
        """ Remove background """
        self.background = None  
        
    
    def set_window(self, img, circleRadius, skinThickness, **kwargs):
        """ Sets the window used for pre and post processing. Use shape = 'circle' or shape = 'square' """
        shape = kwargs.get('shape', 'circle')
        if shape == 'circle':
            self.window = circ_cosine_window(np.shape(img)[0], circleRadius, skinThickness)
        elif shape == 'square':
            self.window = square_cosine_window(np.shape(img)[0], circleRadius, skinThickness)
            
    
    def set_window_radius(self, windowRadius):
        """ Sets the radius of the cropping window """
        self.windowRadius = windowRadius
        
        
    def set_window_thickness(self, windowThickness): 
        """ Sets the edge thickness of the cropping window """
        self.windowThickness = windowThickness

        
    def set_off_axis_mod(self, cropCentre, cropRadius):
        """ Sets the location of the frequency domain position of the OA modulation """
        self.cropCentre = cropCentre
        self.cropRadius = cropRadius
        
        
    def set_stable_ROI(self, roi):
        """ Set the location of the the ROI used for maintaining a constant background phase, i.e.
        this should be a background region of the image. The roi should be an instance of the Roi class.
        """
        self.stableROI = roi

        
    def auto_find_off_axis_mod(self):
        """ Detect the modulation location in frequency domain. """
        if self.background is not None:
            self.cropCentre = off_axis_find_mod(self.background)
            self.cropRadius = off_axis_find_crop_radius(self.background) 
            
    
    def off_axis_background_field(self):
        """ Demodulate the background hologram """
        self.backgroundField = off_axis_demod(self.background, self.cropCentre, self.cropRadius)
        
            
    def update_propagator(self, img):
        """ Create or re-create the propagator using current parameters."""
        self.propagatorWavelength = self.wavelength
        self.propagatorDepth = self.depth

        if self.mode == self.INLINE_MODE:
            if numbaAvailable and self.useNumba:
                self.propagator = propagator_numba(int(np.shape(img)[0] / self.downsample), self.wavelength, self.pixelSize * self.downsample, self.depth)
            else:
                self.propagator = propagator(int(np.shape(img)[0] / self.downsample), self.wavelength, self.pixelSize * self.downsample, self.depth)
            self.propagatorPixelSize = self.pixelSize
        elif self.mode == self.OFFAXIS_MODE:
            if numbaAvailable and self.useNumba:
                self.propagator = propagator_numba(int(np.shape(img)[0] / self.downsample), self.wavelength, self.oaPixelSize * self.downsample, self.depth)
            else:
                self.propagator = propagator(int(np.shape(img)[0] / self.downsample), self.wavelength, self.oaPixelSize * self.downsample, self.depth)
            self.propagatorPixelSize = self.oaPixelSize
     
        
    def set_oa_centre(self, centre):
        """ Set the location of the modulation frequency in frequency domain. """
        self.cropCentre = centre        
     
        
    def set_oa_radius(self, radius):
        """ Set the size of the region to extract in frequency domain to demodulate. """
        self.cropRadius = radius
        
        
    def set_return_FFT(self, returnFFT):
        """ Sets whether the FFT, rather than the demodualted image, is returned in OAH. 
        Set True to obtain FFT, False to obtain image. 
        """        
        self.returnFFT = returnFFT
        
        
    def set_downsample(self, downsample):
        """ Set the downsample factor. This will cause the propagator to be
        recreated when next needed, call update_propagator to force this immediately.
        """
        self.downsample = downsample
        self.propagator = None  # Force to be recreated when needed
        
      
    def process(self, img):
        """ Process an image using the currently selected options """
        #print(self.wavelength, self.pixelSize)
        if img is None: 
            return
        
        ################ INLINE HOLOGRAPHY PIPELINE #######################
        if self.mode is self.INLINE_MODE:
            if self.propagator is None or self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.pixelSize:
                self.update_propagator(img)
            
            # Apply downsampling
            if self.downsample != 1:                
                imgScaled = cv.resize(img, (int(np.shape(img)[1]/ self.downsample), int(np.shape(img)[0] / self.downsample) )   )
            else:
                imgScaled = img
                
            if self.background is not None:
                backgroundScaled = cv.resize(self.background, (int(np.shape(self.background)[1]/ self.downsample), int(np.shape(self.background)[0] / self.downsample)))    
            else:
                backgroundScaled = None                                 
            
            if np.shape(self.propagator) != np.shape(img):
                self.update_propagator(img)
                
            if self.window is not None:
                if np.shape(self.window) != np.shape(imgScaled):
                    if self.windowRadius is None:
                        self.windowRadius = np.shape(img)[0] / 2
                    self.set_window(imgScaled, self.windowRadius / self.downsample, self.windowThickness / self.downsample)
            imgScaled = pre_process(imgScaled, window = self.window, background = backgroundScaled)
            imgOut = refocus(imgScaled, self.propagator, cuda = (self.cuda and cudaAvailable))
            if imgOut is not None:
                imgOut = post_process(imgOut, window = self.window)      
                if self.invert is True:
                    imgOut = np.max(imgOut) - imgOut
                return imgOut
            else:
                return None
        
        
    
        ################# OFF AXIS HOLOGRAPHY PIPELINE ############### 
        if self.mode is self.OFFAXIS_MODE:
                           
            if self.cropCentre is not None and self.cropRadius is not None:
                ret = off_axis_demod(img, self.cropCentre, self.cropRadius, returnFFT = self.returnFFT, cuda = self.cuda)
                if self.returnFFT:
                    demod = ret[0]
                    FFT = ret[1]
                else:
                    demod = ret
            else:
                demod, FFT = None
                return
           
            if self.returnFFT:
                return FFT
            
            if self.relativePhase == True and self.backgroundField is not None:
                demod = relative_phase(demod, self.backgroundField)
            
            if self.refocus is True:
                
                # If we are doing OA we have changed the pixel size in the demodulation process.
                # so here we calculate the corrected pizel size
                self.oaPixelSize = self.pixelSize / float(np.shape(demod)[0]) * float(np.shape(img)[0])

                # Check the propagator is valid, otherwise recreate it
                if self.propagator is None or self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.oaPixelSize:
                    self.update_propagator(demod)
                if np.shape(self.propagator) != np.shape(demod):
                    self.update_propagator(demod)
                
                # Apply the window if requested
                if self.window is not None or self.applyWindow is True:
                    if np.shape(self.window) != np.shape(demod):
                        if self.windowRadius is not None and self.windowThickness is not None:
                            self.set_window(demod, self.windowRadius, self.windowThickness)
                
                # Remove background if requested
                if self.backgroundField is not None:
                    background = np.abs(self.backgroundField)
                    background= None
                else:
                    background = None
                    
                demod = pre_process(demod, window = self.window, background = background)  # background is done elsewhere
                demod = refocus(demod, self.propagator, cuda = (self.cuda and cudaAvailable))
                
                if demod is not None:
                    demod = post_process(demod, window = self.window)     
                
            return demod 
    
        
    def set_use_cuda(self, useCuda):
        """ Set whether to use GPU if available, useCuda is True to use GPU or False to not use GPU.
        """
        self.cuda = useCuda
        
    def set_use_numba(self, useNumba):
        """ Set whether to use Numba JIT if available, useNumba is True to use Numba or False to not use Numba.
        """
        self.useNumba = useNumba    
        
        
        
    def set_find_focus_parameters(self, **kwargs):
        """ Sets the parameters used by the find_focus method """
        self.findFocusDepthRange = kwargs.get('depthRange', (0,0.1))
        self.findFocusRoi = kwargs.get('roi', None)
        self.findFocusMethod = kwargs.get('method', 'Brenner')
        self.findFocusMargin = kwargs.get('margin', None)
        self.coarseSearchInterval = kwargs.get('coarseSearchInterval', None)       
              
        
    def make_propagator_LUT(self, img, depthRange, nDepths):
        """ Creates a LUT of propagators for faster finding of focus """
        self.propagatorLUT = PropLUT(np.shape(img)[0], self.wavelength, self.pixelSize, depthRange, nDepths, numba = (numbaAvailable and self.useNumba))
     
        
    def clear_propagator_LUT(self):
        """ Deletes the LUT of propagators """
        self.propagatorLUT = None
        
        
    def find_focus(self, img):    
        """ Automatically finds the best focus position using defined parameters"""
        args = {"background": self.background,
                "window": self.window,
                "roi": self.findFocusRoi,
                "margin": self.findFocusMargin,
                "numba": numbaAvailable and self.useNumba,
                "cuda": cudaAvailable and self.cuda,
                "propagatorLUT": self.propagatorLUT,
                "coarseSearchInterval": self.findFocusCoarseSearchInterval}
        
        
        return find_focus(img, self.wavelength, self.pixelSize, self.findFocusDepthRange, self.findFocusMethod, **args)
    

    def depth_stack(self, img, depthRange, nDepths):
        """ Create a depth stack using current parameters, producing a set of 
        'nDepths' refocused images. 'depthRange' is a tuple of (min depth, max depth)
        """
        
        if self.mode == self.INLINE_MODE:
            preBackground = self.background
            postBackground = None
        else:            
            preBackground = None
            postBackground = None
        args = {"background": self.background,
                "window": self.window,
                "numba": numbaAvailable and self.useNumba}
                
        return refocus_stack(img, self.wavelength, self.pixelSize, depthRange, nDepths, **args)


    def apply_window(self, img):
        """ Applies the current window to a hologram 'img' """        
        img = pre_process(img, window = self.window)
        return img
    
    
    def auto_focus(self, img, **kwargs):
        """ Customisable auto-focus """
        focusDepth = find_focus(img, self.wavelength, self.pixelSize, 
                               kwargs.get('depthRange',  (0,1) ), \
                               kwargs.get('method', 'Brenner'),  \
                               background = self.background,  \
                               window = self.window,  \
                               numba = self.useNumba and numbaAvailable, \
                               cuda = self.cuda and cudaAvailable, \
                               margin = kwargs.get('margin', None), \
                               propLUT = kwargs.get('propagatorLUT', None),  \
                               coarseSearchInterval = kwargs.get('coarseSearchInterval', None), \
                               roi = kwargs.get('roi', None) ) 
        return focusDepth