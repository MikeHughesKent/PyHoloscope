## -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy in Python

The Holo Class provides an object-oriented interface to mosst of the 
PyHoloscope functionality.

@author: Mike Hughes, Applied Optics Group, Physics & Astronomy, University of Kent
"""

import numpy as np
import cv2 as cv
import time
import warnings

from pyholoscope.utils import circ_cosine_window, square_cosine_window, dimensions
from pyholoscope.off_axis import off_axis_find_mod, off_axis_find_crop_radius, off_axis_demod
from pyholoscope.focusing import propagator, refocus, find_focus, refocus_stack
from pyholoscope.general import pre_process, relative_phase
from pyholoscope.prop_lut import PropLUT


# Check if cupy is available
try:
    import cupy as cp
    cudaAvailable = True
except:
    cudaAvailable = False
    
# Check if numba is available 
try:
    import numba
    from pyholoscope.focusing_numba import propagator_numba
    numbaAvailable = True
    testProg = propagator_numba(int(6),float(1.0),float(1.0),float(1.0), precision = 'single')   # Run the JIT once for speed
except:
    numbaAvailable = False    
   

class Holo:
     
    # Processing pipeline
    INLINE = 1
    OFF_AXIS = 2
    INLINE_MODE = 1   # deprecated, kept for backwards compatibility
    OFFAXIS_MODE = 2  # deprecated, kept for backwards compatibility
    
    # For off-axis holography, these are generated if needed from the 
    # background and normalisation holograms later
    backgroundField = None
    backgroundAbs = None
    backgroundAngle = None
    normaliseField = None
    normaliseAbs = None
    normaliseAngle = None
    
    # When a propagator is created we store the parameters to know when it
    # needs updating
    propagatorDepth = 0
    propagatorWavelength = 0
    propagatorPixelSize = 0
    propagatorSize = 0
    propagator = None
    propagatorLUT = None
    
    # Standard image type
    imageType = 'float32'
    
    def __init__(self, mode = None, wavelength = None, pixelSize = None, **kwargs):
        
        self.mode = mode
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        self.oaPixelSize = pixelSize
        
        self.useNumba = kwargs.get('numba', True)
        self.cuda = kwargs.get('cuda', True)
        
        # Numerical refocusing
        self.depth = kwargs.get('depth', 0)        
        self.set_background(kwargs.get('background', None))
        self.set_normalise(kwargs.get('normalise', None))
        
        # Phase
        self.relativeAmplitude = kwargs.get('relativeAmplitude', None)
        
        # Widowing
        self.autoWindow = kwargs.get('autoWindow', False)
        self.postWindow = kwargs.get('postWindow', False)
        self.window = kwargs.get('window', None) 
        self.windowShape = kwargs.get('windowShape', 'square')        
        self.windowRadius = kwargs.get('windowRadius', None)
        self.windowThickness = kwargs.get('windowThickness', 10)
        
        # Autofocus
        self.findFocusMethod = kwargs.get('findFocusMethod', 'Brenner')
        self.findFocusRoi = kwargs.get('findFocusRoi', None)
        self.findFocusMargin = kwargs.get('findFocusMargin', None)
        self.findFocusCoarseSearchInterval = kwargs.get('findFocusCoarseSearchInterval', None)
        self.findFocusDepthRange = kwargs.get('findFocusDepthRange', (0,1))

        # Off-axis demodulation
        self.cropCentre = kwargs.get('cropCentre', False)
        self.cropRadius = kwargs.get('cropRadius', False)
        self.returnFFT = kwargs.get('returnFFT', False)
        
        
        # Phase
        self.relativePhase = kwargs.get('relativePhase', False)
        self.stablePhase = kwargs.get('stablePhase', False)
        self.stableROI = kwargs.get('stableROI', False)
        
        # Display
        self.invert = kwargs.get('invert', False)
        self.refocus = kwargs.get('refocus', False)
        self.downsample = kwargs.get('downsample', 1)        
       
        # GPU
        self.cudaAvailable = cudaAvailable
        
        # Image data type
        self.precision = self.set_precision(kwargs.get('precision', 'single'))
    
        
    def process(self, img):
        """ Process a hologram using the currently selected parameters. 
        Calls process_inline or process_off_axis depending  on mode.
        """
      
        # If we are refocusing we must have a wavelength, pixel size and depth specified
        if self.mode == self.INLINE or (self.mode == self.OFF_AXIS and self.refocus == True):
            assert self.pixelSize is not None, "Pixel size not specified."
            assert self.wavelength is not None, "Wavelength not specified."
            assert self.depth is not None, "Refocus depth not specified."
        
        if img is None: 
            warnings.warn('Image provided to process was None, output will be None.')
            return None
        
        assert img.ndim == 2, "Input must be a 2D numpy array."        
        
        if self.mode == self.INLINE_MODE or self.mode == self.INLINE: 
            return self.process_inline(img)
        elif self.mode == self.OFFAXIS_MODE or self.mode == self.OFF_AXIS: 
            return self.process_off_axis(img) 
        else: 
            raise Exception("Invalid processing mode.")
        
        
        
    def process_inline(self, img):
        """ Process an  inline hologram image, img, using the currently selected 
        parameters.
        """
        
        # If we have are doing autowindow, and we either don't have
        # a window, or it is the wrong size, we make a new window   
        self.update_auto_window(img)
                   
        # Apply background, normalisation, windowing, downsampling    
        if self.normaliseField is not None:
            normalise = np.abs(self.normaliseField)
        else:
            normalise = None
        imgPreprocessed = pre_process(img, downsample = self.downsample, window = self.window, background = self.background, normalise = self.normalise, precision = self.precision)
      
        
        # If the propagator is not the correct one for the current parameters, regenerate it
        if np.shape(self.propagator) != np.shape(imgPreprocessed) or self.propagator is None or self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.pixelSize * self.downsample:
            self.update_propagator(img)

        # Numerical refocusing           
        imgOut = refocus(imgPreprocessed, self.propagator, cuda = (self.cuda and cudaAvailable))
        
        if imgOut is None:
            warnings.warn('Output from refocusing was None.')
            return None
            
        # Post refocusing processing
        if self.postWindow is True and self.window is not None:
            imgOut = pre_process(imgOut, window = self.window, precision = self.precision)  
                               
        if self.invert is True:
            imgOut = np.max(imgOut) - imgOut
        
        return imgOut
        
   
    def process_off_axis(self, img):
        """ Process an off-axis hologram image using the currently selected parameters.
        """
                        
        assert self.cropCentre is not None, "Off-Axis demodulation frequency not defined."
        assert self.cropRadius is not None, "Off-Axis demodulation radius not defined."
          
        # Removes the off-axis modulation to obtain complex image
        demod = off_axis_demod(img, self.cropCentre, self.cropRadius, returnFFT = self.returnFFT, cuda = self.cuda)
        
        if demod is None:
            warnings.warn('Output from off-axis demodulation was None.')
            return None  
            
        # If returnFFT is True, off_axis_demod returns the demodulated image and the FFT as a tuple. IF we
        # have been asked for the FFT we pull this out and return it, otherwise 'demod' is the demodulated images
        # and we continue
        if self.returnFFT:
            return demod[1]        
        
        # Relative phase means to subtract the phase from the background image
        if self.relativePhase == True:
            if self.backgroundField is not None:
                demod = relative_phase(demod, self.backgroundField)
            else:
                warnings.warn('Relative phase requested but no background field available, call off_axis_background_field() to create this first.')
 
        if demod is None:
            warnings.warn('Output from off-axis relative phase was None.')
            return None
    
        # If we have are doing autowindow, and we either don't have
        # a window, or it is the wrong size, we make a new window   
        self.update_auto_window(img)        
           
        # Off axis demodulation changes the pixel size,
        # so here we calculate the corrected pizel size
        if self.pixelSize is not None:
            self.oaPixelSize = self.pixelSize / float(np.shape(demod)[0]) * float(np.shape(img)[0])
                        
        # Apply background, normalisation, windowing, downsampling    
        if self.relativeAmplitude:
            background = self.backgroundAbs
        else:
            background = None
        demod = pre_process(demod, downsample = self.downsample, window = self.window, background = background, normalise = self.normaliseAbs, precision = self.precision)
    
        # Numerical refocusing
        if self.refocus is True:
 
            # Check the propagator is valid, otherwise recreate it
            if np.shape(self.propagator) != np.shape(demod) or self.propagator is None or self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.oaPixelSize:
                self.update_propagator(demod)
           
            # Refocus
            demod = refocus(demod, self.propagator, cuda = (self.cuda and cudaAvailable))
            
            if demod is None:
                warnings.warn('Output from off-axis refocusing was None.')
                return None
            
        # Post refocusing processing
        if demod is not None:
            if self.postWindow is True and self.autoWindow is True and self.window is not None:
                demod = pre_process(demod, window = self.window, precision = self.precision)  
                                   
            if self.invert is True: demod = np.max(demod) - demod
            
        if demod is None:
            warnings.warn('Output from off-axis processing was None.')
            return None

        return demod
            
    
    
    def __str__(self):
        return "PyHoloscope Holo Class. Wavelength: " + str(self.wavelength) + ", Pixel Size: " + str(self.pixelSize)


    def set_depth(self, depth):
        """ Set the depth for numerical refocusing 
        """
        self.depth = depth

        
    def set_wavelength(self, wavelength):
        """ Set the wavelength of the hologram 
        """
        self.wavelength = wavelength

        
    def set_pixel_size(self, pixelSize):
        """ Set the size of pixels in the raw hologram
        """
        self.pixelSize = pixelSize     
        
        
    def set_background(self, background):
        """ Set the background hologram. Use None to remove background. 
        """
        self.clear_background()
        if background is not None: self.background  = background.astype(self.imageType) 
      
            
    def set_normalise(self, normalise):
        """ Set the normalisation hologram. Use None to remove normalisation. 
        """
        self.clear_normalise()     
        if normalise is not None:
            self.normalise  = normalise.astype(self.imageType) 
       
            
    def clear_background(self):
        """ Remove existing background hologram. 
        """
        self.background = None 
        self.backgroundField = None
        self.backgroundAbs = None
        self.backgroundAngle = None
                
    
    def clear_normalise(self):
        """ Remove existing normalisation hologram. 
        """
        self.normalise = None      
        self.normaliseField = None
        self.normaliseAbs = None
        self.normaliseAngle = None
     
         
    def set_relative_amplitude(self, boolean):
         """ Sets whether or not to calculate relative phase in off-axis holography.
         """
         assert boolean == True or boolean == False, "Argument of set_relative_amplitude must be True or False"
         self.relativeAmplitude = boolean 
        
    
    def set_relative_phase(self, boolean):
         """ Sets whether or not calculate relative amplitude in off-axis holography.
         """
         assert boolean == True or boolean == False, "Argument of set_relative_phase must be True or False"
         self.relativePhase = boolean
         
    
    def set_precision(self, precision):
        """ Sets whether to use single or double precision.
        """
        assert (precision == 'single' or precision == 'double'), "Precision must be 'single' or 'double'."
        self.precision = precision
        if self.precision == 'double':
            self.imType = 'float64'
        else:
            self.imType = 'float32'
            
            
    def create_window(self, imgSize, radius, skinThickness, shape = 'square'):
        """ Creates and stores the window used for pre or post processing. 
        
        Arguments:
            imgSize       :  the size of the window array, must be the same as the hologram it will be
                             applied to. Either provide a 2D numpy array, in which case the window will 
                             be created to match the size of this, provide an int, in which case the window 
                             will be a square of this size or a tuple of (width, height).
            radius        :  the size of the  transparent part of the window, for 'circle' this is the 
                             radius, for 'square' this is half the side length. For 'circle' provide
                             an int, for 'square' either provide an int (resulting in a square window)
                             or a tuple of (width, height) for rectangular window.
            skinThickness :  The number of pixels inside the window over which it transitions from
                             opaque to transparent.
                             
        Keyword Arguments:                     
            shape         :  [Optional] window shape, 'circle' or 'square' (defualt).
        """    

        if shape == 'circle':
            self.window = circ_cosine_window(imgSize, radius, skinThickness, dataType = self.imageType)
        elif shape == 'square':
            self.window = square_cosine_window(imgSize, radius, skinThickness, dataType = self.imageType)
    
    
    def set_window(self, window):
        """ Sets the window to a pre-generated 'window', a 2D numpy array.
        """
        self.clear_window()
        if window is not None: 
            self.window = window.astype(self.imageType)       
    
    
    def set_window_shape(self, windowShape):
        """ Sets the window shape, 'cicle' or 'square'.
        """
        if windowShape == 'circle' or windowShape == 'square':
            self.windowShape = windowShape
        else:
            raise Exception ("Invalid window shape.")
        
            
    def clear_window(self):
        """ Removes existing window, equivalent to set_window(None)
        """
        self.window = None
        
        
    def set_auto_window(self, autoWindow):
        """ Sets whether or not use auto create a window (boolean).
        """
        assert autoWindow == True or autoWindow == False, "set_auto_window must be True or Falses"
        self.autoWindow = autoWindow
        
        
    def set_post_window(self, postWindow):
        """ Sets whether or not to re- apply the window after refocusing (boolean).
        """
        assert postWindow == True or postWindow == False, "set_post_window must be True or False"
        self.postWindow = postWindow    
        
        
    def set_window_radius(self, windowRadius):
        """ Sets the radius of the cropping window.
        """
        self.windowRadius = windowRadius
        
        
    def set_window_thickness(self, windowThickness): 
        """ Sets the edge thickness of the cropping window.
        """
        self.windowThickness = windowThickness


    def update_auto_window(self, img):
        """ Create or re-create the automatic window using current parameters.
        Provide an 'img', a 2D numpy array of the same size as the image to
        be processed.
        """

        imHeight = np.shape(img)[0]
        imWidth = np.shape(img)[1]
        
        if self.autoWindow == True:
            if self.window is None:
                regenWindow = True
            elif np.shape(self.window)[0] != np.shape(img)[0] / self.downsample or np.shape(self.window)[1] != np.shape(img)[1] / self.downsample:
                regenWindow = True
            else: 
                regenWindow = False
                
            if regenWindow:
               if self.windowRadius is None:
                   windowRadiusX = int(imWidth / 2)
                   windowRadiusY = int(imHeight / 2)
               else:
                   windowRadiusX, windowRadiusY = dimensions(self.windowRadius)
                   
               self.create_window( (int(imWidth / self.downsample), int(imHeight / self.downsample)), 
                                   (int(windowRadiusX / self.downsample), int(windowRadiusY / self.downsample) ),
                                   self.windowThickness / self.downsample,
                                   shape = self.windowShape)

    def set_off_axis_mod(self, cropCentre, cropRadius):
        """ Sets the location of the frequency domain position of the OA modulation.
        
        Arguments:
            cropCentre  : tuple of (x,y)
            cropRadius  : radius
        """
        self.cropCentre = cropCentre
        self.cropRadius = cropRadius
        
        
    def set_stable_ROI(self, roi):
        """ Set the location of the the ROI used for maintaining a constant 
        background phase, i.e. this should be a background region of the image. 
        The roi should be an instance of the Roi class.
        """
        self.stableROI = roi

        
    def auto_find_off_axis_mod(self, maskFraction = 0.1):
        """ Detect the modulation location in frequency domain. maskFraction
        is the size of a mask applied to the centre of the FFT to prevent
        the d.c. from being detected. 
        """
        if self.background is not None:
            self.cropCentre = off_axis_find_mod(self.background, maskFraction = 0.1)
            self.cropRadius = off_axis_find_crop_radius(self.background) 
            
    
    def calib_off_axis(self, hologram = None, maskFraction = 0.1):
        """ Detect the modulation location in frequency domain using the 
        background or a provided hologram. 
        """

        if hologram is None:
            hologram = self.background
        if hologram is None:
            raise Exception ("calib_off_axis requires a calibration image, either provided as an argument or from a previously set background.")
        self.cropCentre = off_axis_find_mod(hologram, maskFraction = 0.1)
        self.cropRadius = off_axis_find_crop_radius(hologram) 
        if self.background is not None:
            self.off_axis_background_field()
        if self.normalise is not None:
            self.off_axis_normalise_field()
    
    
    def off_axis_background_field(self):
        """ Demodulate the background hologram.
        """
        assert self.background is not None, 'Background hologram not provided.'
        assert self.cropCentre is not None, 'Demodulation centre not provided'       
        assert self.cropRadius is not None, 'Demodulation radius not provided.'
        self.backgroundField = off_axis_demod(self.background, self.cropCentre, self.cropRadius)
        self.backgroundAbs = np.abs(self.backgroundField)   # Store these now for speed
        self.backgroundPhase = np.angle(self.backgroundField)

    def off_axis_normalise_field(self):
        """ Demodulate the background hologram.
        """
        assert self.background is not None, 'Background hologram not provided.'
        assert self.cropCentre is not None, 'Demodulation centre not provided'       
        assert self.cropRadius is not None, 'Demodulation radius not provided.'
        self.normaliseField = off_axis_demod(self.normalise, self.cropCentre, self.cropRadius)
        self.normaliseAbs = np.abs(self.normaliseField)      # Store these now for speed
        self.normalisePhase = np.angle(self.normaliseField)
        
            
    def update_propagator(self, img):
        """ Create or re-create the propagator using current parameters. img 
        should be an 2D numpy array of the same size as the images to be processed.
        """
        self.propagatorWavelength = self.wavelength
        self.propagatorDepth = self.depth

        if self.mode == self.INLINE_MODE: 
            self.propagatorPixelSize = self.pixelSize * self.downsample
        else:
            self.propagatorPixelSize = self.oaPixelSize * self.downsample
        
        propWidth = int(np.shape(img)[1] / self.downsample / 2 ) * 2   
        propHeight = int(np.shape(img)[0] / self.downsample / 2) * 2    
        
        if numbaAvailable and self.useNumba:
            self.propagator = propagator_numba((propWidth, propHeight), self.wavelength, self.propagatorPixelSize, self.depth, precision = self.precision)
        else:
            self.propagator = propagator((propWidth, propHeight), self.wavelength, self.propagatorPixelSize, self.depth, precision = self.precision)
     
        # If using CUDA we send propagator to GPU now to speed up refocusing later 
        if self.cuda and cudaAvailable:
           self.propagator = cp.array(self.propagator)
    
    
           
    def set_oa_centre(self, centre):
        """ Set the location of the modulation frequency in frequency domain. 
        'centre' is a tuple of (x,y).
        """
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
        if downsample != self.downsample:
            self.propagator = None  # Force to be recreated when needed
            
        self.downsample = downsample
        
        
    def set_use_cuda(self, useCuda):
        """ Set whether to use GPU if available, useCuda is True to use GPU or 
        False to not use GPU.
        """
        self.cuda = useCuda
        
        
    def set_use_numba(self, useNumba):
        """ Set whether to use Numba JIT if available, useNumba is True to use 
        Numba or False to not use Numba.
        """
        self.useNumba = useNumba    
                
        
    def set_find_focus_parameters(self, **kwargs):
        """ Sets the parameters used by the find_focus method.
        
        Keyword Arguments:
            depthRange   : double
                           tuple of (min, max) depths to search within in m.
            roi          : instance of Roi
                           area to assess focus within, default is None in which
                           case all of image is used.
            method       : str
                           focus metric to use.
            margin       : int
                           if specified only the Roi and a margin will be 
                           refocused. If None (default) the whole image will be
                           refocused regardless. Has no effect if roi not specified.
            coarseSearchInterval  : Number of points to check explicitly before
                                    optimising. Default is None, in which case
                                    this is not performed.
        
        
        """
        self.findFocusDepthRange = kwargs.get('depthRange', (0,0.1))
        self.findFocusRoi = kwargs.get('roi', None)
        self.findFocusMethod = kwargs.get('method', 'Brenner')
        self.findFocusMargin = kwargs.get('margin', None)
        self.coarseSearchInterval = kwargs.get('coarseSearchInterval', None)       
              
        
    def make_propagator_LUT(self, img, depthRange, nDepths):
        """ Creates a LUT of propagators for faster finding of focus.
        """
        self.propagatorLUT = PropLUT(np.shape(img)[0], self.wavelength, self.pixelSize, depthRange, nDepths, numba = (numbaAvailable and self.useNumba), precision = self.precision)
     
        
    def clear_propagator_LUT(self):
        """ Deletes the LUT of propagators.
        """
        self.propagatorLUT = None
        
        
    def find_focus(self, img):    
        """ Automatically finds the best focus position for hologram 'img' 
        using defined parameters.
        """
        
        args = {"background": self.background,
                "window": self.window,
                "roi": self.findFocusRoi,
                "margin": self.findFocusMargin,
                "numba": numbaAvailable and self.useNumba,
                "cuda": cudaAvailable and self.cuda,
                "propagatorLUT": self.propagatorLUT,
                "coarseSearchInterval": self.findFocusCoarseSearchInterval}
        
        
        return find_focus(img, self.wavelength, self.pixelSize, self.findFocusDepthRange, self.findFocusMethod, precision = self.precision, **args)
    

    def depth_stack(self, img, depthRange, nDepths):
        """ Create a depth stack using current parameters, producing a set of 
        'nDepths' refocused images equally spaced within depthRange. 
        Arguments:
            img        : ndarray
                         hologram 
            depthRange : tuple 
                         depths to focus to: (min depth, max depth)
            nDepths    : int
                         number of depths to create images for within depthRange
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
                
        return refocus_stack(img, self.wavelength, self.pixelSize, depthRange, nDepths, precision = self.precision, **args)


    def apply_window(self, img):
        """ Applies the current window to a hologram 'img'.
        """        
        img = pre_process(img, window = self.window)
        return img
    
    
    def auto_focus(self, img, **kwargs):
        """ Customisable auto-focus.
        """
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
                               roi = kwargs.get('roi', None) ,
                               precision = self.precision) 
        return focusDepth