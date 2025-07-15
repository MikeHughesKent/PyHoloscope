# -*- coding: utf-8 -*-
"""
PyHoloscope: focus_stack

Class to store stack of images numerically refocused to different depths.

"""

import math

import numpy as np
from PIL import Image


################### Class for stack of images focused at different depths ####       
class FocusStack:
     
    def __init__(self, img, depthRange, nDepths):
        """ Initialise stack.
        
        Arguments:
            img        : numpy.ndarray
                         example image of the correct size and type, 2D numpy array
            depthRange : (float, float)
                         tuple of min depth and max depth in stack
            nDepths    : int
                         number of images to be stored in stack
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
        """ Adds an image to a specific index position.
        
        Arguments:
            img      : ndarray
                       image as a 2D numpy array
            idx      : int
                       index position                       
        """
        self.stack[idx, :,:] = img  
        
        
    def add_depth(self, img, depth):
        """ Adds an image to index position closest to the the specifed depth.
        
        Arguments:
            img      : ndarray
                       image as a 2D numpy array
            depth    : float
                       refocus depth of image         
        """
        self.stack[self.depth_to_index(depth),:,:] = img
    
        
    def get_index(self, idx):
        """ Returns the refocused image stored at the specified index.
        
        Arguments:
            idx      : int
                       index position to return image from
                       
        Returns:
            numpy.ndarray : image               
        
        """
        return self.stack[idx, : , :]
    
    
    def get_depth(self, depth):
        """ Returns the closest refocused image to the specifed depth.
        
        Arguments:
            depth     : float
                        depth to return image from
                        
        Returns:
            numpy.ndarray : image                
        
        """
        return self.get_index(self.depth_to_index(depth))
    
    
    def get_depth_intensity(self, depth):
        """ Returns the amplitude of the refocused image closest to the specified depth.
        
        Parameters:
            depth     : float
                        depth to return image from
                        
        Returns:
            numpy.ndarray : image                
                
        """
        return np.abs(self.get_depth(depth))

    
    def get_index_intensity(self, idx):
        """ Return the amplitude of the refocused image at the specified index.
        
        Parameters:
            idx     : int
                      index position to return image from
                        
        Returns:
            numpy.ndarray : image                
                
        """        
        return np.abs(self.get_index(idx))
    
    
    def depth_to_index(self, depth):
        """ Returns the index closest to the specified depth.
        
        Parameters:
            depth     : float
                        depth to obtain closest index to
                        
        Returns:
            int       : index                
                
        """ 
        idx = round((depth - self.minDepth) / (self.maxDepth - self.minDepth) * (self.nDepths-1))
        if idx < 0:
            idx = 0
        if idx > self.nDepths - 1:
            idx = self.nDepths - 1
        return idx
    
    
    def index_to_depth(self, idx):
        """ Returns depth corresponding to the specified index.
        
        Parameters:
            idx     : int
                      index position
                        
        Returns:
            float   : depth           
                
        """ 
        return self.depths[idx]
                           
    
    def write_intensity_to_tif(self, filename, autoContrast = True):
        """ Writes the amplitudes of the stack of refocused images to a 16 bit tif stack 
        If autoContrast == True, all images will be autoscaled (across the whole
        stack, not individually) to use the full bit depth.
        
        Arguments:
            filename     : str
                           path/file to save to, should have .tif extension.
                           
        Keyword Arguments:
            autoContrast : boolean
                           if True (default) images are autoscaled         
        """
       
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
        """ Writes the phases of the stack of refocused images to a 16 bit tif stack.
        -pi is mapped to 0, pi is mapped to 255.
        
        Arguments:
            filename     : str
                           path/file to save to, should have .tif extension.
        """

        imlist = []
        for m in self.stack:
            im = (np.angle(m) + math.pi) * 255
            imlist.append(Image.fromarray(im.astype('uint16')))

        imlist[0].save(filename, compression="tiff_deflate", save_all=True,
               append_images=imlist[1:])
        
