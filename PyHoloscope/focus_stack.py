# -*- coding: utf-8 -*-
"""
PyHoloscope: focus_stack


Class for stack of images numerically refocused to different depths.


@author: Mike Hughes
Applied Optics Group, Physics & Astronomy, University of Kent
"""

import numpy as np
from PIL import Image
import math


################### Class for stack of images focused at different depths ####       
class FocusStack:
     
    def __init__(self, img, depthRange, nDepths):
        self.stack = np.zeros((nDepths, np.shape(img)[0], np.shape(img)[1]), dtype = 'complex128')
        self.depths = np.linspace(depthRange[0], depthRange[1], nDepths)
        self.minDepth = depthRange[0]
        self.maxDepth = depthRange[1]
        self.nDepths = nDepths
        self.depthRange = depthRange
        
    def __str__(self):
        return "Refocus stack. Min: " + str(self.minDepth) + ", Max: " + str(self.maxDepth) + ", Num: " + str(self.nDepths) + ", Step: " + str((self.maxDepth - self.minDepth) / self.nDepths)
        
    def addIdx(self, img, idx):
        self.stack[idx, :,:] = img        
        
    def addDepth(self, img, depth):
        self.stack[self.depthToIndex(depth),:,:] = img
        
    def getIndex(self, idx):
        #print("Getting image at index ", idx)
        #print(self.stack[idx, : , :])
        return self.stack[idx, : , :]
    
    def getDepth(self, depth):
        #print("Getting image from depth ", depth)
        return self.getIndex(self.depthToIndex(depth))
        
    def getDepthIntensity(self, depth):
        #print("Getting image intensity from depth ", depth)
        #print(self.getDepth(depth))
        return np.abs(self.getDepth(depth))
    
    def getIndexIntensity(self, idx):
        #print("Getting image intensity at index ", idx)
        return np.abs(self.getIndex(idx))
    
    def depthToIndex(self, depth):
        idx = round((depth - self.minDepth) / (self.maxDepth - self.minDepth) * self.nDepths)
        if idx < 0:
            idx = 0
        if idx > self.nDepths - 1:
            idx = self.nDepths - 1
        return idx
    
    def writeIntensityToTif(self, filename, **kwargs):
        
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
        
    def writePhaseToTif(self, filename):
        imlist = []
        for m in self.stack:
            im = (np.angle(m) + math.pi) * 255
            imlist.append(Image.fromarray(im.astype('uint16')))

        imlist[0].save(filename, compression="tiff_deflate", save_all=True,
               append_images=imlist[1:])
        
