## -*- coding: utf-8 -*-
"""
PyHoloscope.holo_class

The Holo Class provides an object-oriented interface to a subset of the 
PyHoloscope functionality.

@author: Mike Hughes
Applied Optics Group, Physics & Astronomy, University of Kent
"""

from PyHoloscope.general import *

class Holo:
    
    def __init__(self, mode, wavelength, pixelSize, **kwargs):
        
        self.mode = mode
        self.wavelength = wavelength
        self.pixelSize = pixelSize
        
        self.depth = kwargs.get('depth', 0)
        self.background = kwargs.get('background',None)
        self.applyWindow = False
        self.window = kwargs.get('window', None)        
        self.windowRadius = kwargs.get('windowRadius', None)
        self.windowThickness = kwargs.get('windowThickness', None)
        
        self.findFocusMethod = kwargs.get('findFocusMethod', 'Sobel')
        self.findFocusRoi = kwargs.get('findFocusRoi', None)
        self.findFocusMargin = kwargs.get('findFocusMargin', None)
        self.cuda = kwargs.get('cuda', False)
       
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
        self.depth = depth
        
    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
        
    def set_pixel_size(self, pixelSize):
        self.pixelSize = pixelSize        
        
    def set_background(self, background):
        if background is not None:
            self.background  = background.astype('uint16') 
        else:
            self.background = None
        
    def clear_background(self):
        self.background = None        
    
    def set_window(self, img, circleRadius, skinThickness, **kwargs):
        shape = kwargs.get('shape', 'circle')
        if shape == 'circle':
            self.window = circ_cosine_window(np.shape(img)[0], circleRadius, skinThickness)
        elif shape == 'square':
            self.window = square_cosine_window(np.shape(img)[0], circleRadius, skinThickness)
    
    def set_window_radius(self, windowRadius):
        self.windowRadius = windowRadius
        
    def set_window_thickness(self, windowThickness): 
        self.windowThickness = windowThickness
        
    def set_off_axis_mod(self, cropCentre, cropRadius):
        self.cropCentre = cropCentre
        self.cropRadius = cropRadius
        
    def set_stable_ROI(self, roi):
        self.stableROI = roi
       # print(f"holo_class_set_stableROI {roi}")
        
    def auto_find_off_axis_mod(self):
        if self.background is not None:
            self.cropCentre = off_axis_find_mod(self.background)
            self.cropRadius = off_axis_find_crop_radius(self.background)         
    
    def off_axis_background_field(self):
        self.backgroundField = off_axis_demod(self.background, self.cropCentre, self.cropRadius)
                    
    def update_propagator(self, img):
        self.propagator = propagator(int(np.shape(img)[0] / self.downsample), self.wavelength, self.pixelSize * self.downsample, self.depth)
        self.propagatorWavelength = self.wavelength
        self.propagatorPixelSize = self.pixelSize
        self.propagatorDepth = self.depth
        
    def set_oa_centre(self, centre):
        self.cropCentre = centre        
     
    def set_oa_radius(self, radius):
        self.cropRadius = radius
        
    def set_return_FFT(self, returnFFT):
        self.returnFFT = returnFFT
        
    def set_downsample(self, downsample):
        self.downsample = downsample
        self.propagator = None  #Force to be recreated when needed
        
      
    def process(self, img):
        t0 = time.time()
        #print(f"holo_class_process img {img}")
        if img is None: 
            return
        
        ################ INLINE HOLOGRAPHY PIPELINE #######################
        if self.mode is PyHoloscope.INLINE_MODE:
            if self.propagator is None or self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.pixelSize:
                self.update_propagator(img)
            
            # Apply downsampling
            imgScaled = cv.resize(img, (int(np.shape(img)[1]/ self.downsample), int(np.shape(img)[0] / self.downsample) )   )
           
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
            imgOut = refocus(imgScaled, self.propagator, cuda = self.cuda)
            if imgOut is not None:
                imgOut = post_process(imgOut, window = self.window)      
                if self.invert is True:
                    imgOut = np.max(imgOut) - imgOut
                return imgOut
            else:
                return None
        
        
    
        ################# OFF AXIS HOLOGRAPHY PIPELINE ############### 
        if self.mode is PyHoloscope.OFFAXIS_MODE:
            #print("off axis mode")  
                           
            if self.cropCentre is not None and self.cropRadius is not None:
                #t1 = time.perf_counter()
                ret = off_axis_demod(img, self.cropCentre, self.cropRadius, returnFFT = self.returnFFT, cuda = self.cuda)
                #print("demod time", time.perf_counter() - t1)
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
            
            #if self.stablePhase == True:
            #    demod = stable_phase(demod, roi = self.stableROI)
           
            if self.refocus is True:
              
                if self.propagatorDepth != self.depth or self.propagatorWavelength != self.wavelength or self.propagatorPixelSize != self.pixelSize:
                    self.update_propagator(demod)
                
                if np.shape(self.propagator) != np.shape(demod):
                    self.update_propagator(demod)
                
                if self.window is not None or self.applyWindow is True:
                    if np.shape(self.window) != np.shape(demod):
                        if self.windowRadius is not None and self.windowThickness is not None:
                            self.set_window(demod, self.windowRadius, self.windowThickness)
                if self.backgroundField is not None:
                    background = np.abs(self.backgroundField)
                    background= None
                else:
                    background = None
                demod = pre_process(demod, window = self.window, background = background)  # background is done elsewhere
                demod = refocus(demod, self.propagator, cuda = self.cuda)
                
                if demod is not None:
                    demod = post_process(demod, window = self.window)            
                              
                
            return demod 
              
       
        
    def set_find_focus_parameters(self, method, roi, margin, depthRange):
        self.findFocusMethod = method
        self.findFocusRoi = roi
        self.findFocusMargin = margin
        self.findFocusDepthRange = depthRange
        
        
    def make_propagator_LUT(self, img, depthRange, nDepths):
        self.propagatorLUT = PropLUT(np.shape(img)[0], self.wavelength, self.pixelSize, depthRange, nDepths)
     
        
    def clear_propagator_LUT(self):
        self.propagatorLUT = None
        
        
    def find_focus(self, img):        
                
        args = {'background': self.background,
                "window": self.window,
                "roi": self.findFocusRoi,
                "margin": self.findFocusMargin,
                "propagatorLUT": self.propagatorLUT}
        
        return find_focus(img, self.wavelength, self.pixelSize, self.findFocusDepthRange, self.findFocusMethod, **args)
    

    def depth_stack(self, img, depthRange, nDepths):
        
        if self.mode == PyHoloscope.INLINE_MODE:
            background = self.background
        else:
            background = None
        args = {'background': background,
                "window": self.window}
                
        return refocus_stack(img, self.wavelength, self.pixelSize, depthRange, nDepths, **args)

    def off_axis_recon(self, img):
        recon = off_axis_demod(img, self.cropCentre, self.cropRadius)
        if self.relativePhase == True and self.backgroundField is not None:
            recon = relative_phase(recon, self.backgroundField)
        return recon
    
    
    def apply_window(self, img):
        
        img = preProcess(img, window = self.window)
        return img
    
    def auto_focus(self, img, **kwargs):
        t1 = time.perf_counter()
        focusDepth = find_focus(img, self.wavelength, self.pixelSize, 
                               kwargs.get('depthRange',  (100,1000) ), \
                               kwargs.get('method', 'Brenner'),  \
                               background = self.background,  \
                               window = self.window,  \
                               scoreROI = kwargs.get('scoreROI', None), \
                               margin = kwargs.get('margin', None), \
                               propLUT = kwargs.get('propagatorLUT', None),  \
                               coarseSearchInterval = kwargs.get('coarseSearchInterval', None), \
                               roi = kwargs.get('roi', None) ) 
        return focusDepth