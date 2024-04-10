# -*- coding: utf-8 -*-
"""
PyHoloscope: focus_stack

Class to store stack of images numerically refocused to different depths.

@author: Mike Hughes, Applied Optics Group, Physics & Astronomy, University of Kent
"""

import math


import numpy as np
from PIL import Image


################### Class for stack of images focused at different depths ####       
class FocusStack:
     
    def __init__(self, img, depthRange, nDepths):
        """ Initialise stack.
        img : example image of the correct size, 2D numpy array
        depthRange : tuple of min depth and max depth in stack
        nDepths : number of images to be stored in stack
        """
        self.stack = np.zeros((nDepths, np.shape(img)[0], np.shape(img)[1]), dtype = img.dtype)
        self.depths = np.linspace(depthRange[0], depthRange[1], nDepths)
        self.minDepth = depthRange[0]
        self.maxDepth = depthRange[1]
        self.nDepths = nDepths
        self.depthRange = depthRange
        
        
    def __str__(self):
        return "Refocus stack. Min: " + str(self.minDepth) + ", Max: " + str(self.maxDepth) + ", Num: " + str(self.nDepths) + ", Step: " + str((self.maxDepth - self.minDepth) / self.nDepths)
        
    
    def add_idx(self, img, idx):
        """ Add a refocused image to a specific idx"""
        self.stack[idx, :,:] = img  
        
        
    def add_depth(self, img, depth):
        """ Add an image image to index closest to the the specifed depth """
        self.stack[self.depth_to_index(depth),:,:] = img
    
        
    def get_index(self, idx):
        """ Return the refocused image at the specified index """
        return self.stack[idx, : , :]
    
    
    def get_depth(self, depth):
        """ Return the closest refocused image to the specifed depth """
        return self.get_index(self.depth_to_index(depth))
    
    
    def get_depth_intensity(self, depth):
        """ Return the amplitude of the refocused image closest to the specified depth """
        return np.abs(self.get_depth(depth))

    
    def get_index_intensity(self, idx):
        """ Return the amplitude of the refocused image at the specified index """        
        return np.abs(self.get_index(idx))
    
    
    def depth_to_index(self, depth):
        """ Return the index closest to the specified depth """
        
        idx = round((depth - self.minDepth) / (self.maxDepth - self.minDepth) * (self.nDepths-1))
        if idx < 0:
            idx = 0
        if idx > self.nDepths - 1:
            idx = self.nDepths - 1
        return idx
    
    
    def index_to_depth(self, idx):
        """ Return depth corresponding to the specified index """
        return self.depths[idx]
                           
    
    def write_intensity_to_tif(self, filename, **kwargs):
        """ Write the amplitudes of the stack of refocused images to a tif stack 
        If autoContrast == True, all images will be normalised.
        """
        
        autoContrast = kwargs.get('autoContrast', True)
       
        if autoContrast:
           maxVal = np.max(np.abs(self.stack))
           minVal = np.min(np.abs(self.stack))
        
        imlist = []
        for m in self.stack:
            if autoContrast:
                im = np.abs(m).astype('float64') - minVal
                im = im / (maxVal - minVal) * 2**16
                imlist.append(Image.fromarray(im.astype('uint16')))

            else:
                imlist.append(Image.fromarray((255 * np.abs(m)).astype('uint16')))

        imlist[0].save(filename, compression="tiff_deflate", save_all=True,
               append_images=imlist[1:])
        
        
    def write_phase_to_tif(self, filename):
        """ Write the phases of the stack of refocused images to 16 bit tif stack 
        -pi is mapped to 0, pi is mapped to 255.
        """

        imlist = []
        for m in self.stack:
            im = (np.angle(m) + math.pi) * 255
            imlist.append(Image.fromarray(im.astype('uint16')))

        imlist[0].save(filename, compression="tiff_deflate", save_all=True,
               append_images=imlist[1:])
        
