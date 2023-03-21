# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Roi: Class for region of interest.


@author: Mike Hughes
Applied Optics Group, Physics & Astronomy, University of Kent
"""


class Roi:
    
    def __init__(self, x, y, width, height):
        self.x = max(int(x),0)
        self.y = max(int(y),0)
        self.width = max(int(width),0)
        self.height = max(int(height),0)
        
    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.width) + ',' + str(self.height)
    
    def constrain(self, minX, minY, maxX, maxY):
        """ Stop ROI exceeding image size"""
        self.x = max(self.x, minX, 0)
        self.y = max(self.y, minY, 0)

        self.width = max(min(self.width, maxX - self.x), 0)
        self.height = max(min(self.height, maxY - self.y), 0)
        
    def crop (self, img):
        """ img is cropped to ROI  """  
        return img[self.y: self.y + self.height, self.x:self.x + self.width]
    
    def clear_outside(self, img):
        """Set pixels in img to be zero if outside ROI"""
        imgOut = img.copy()
        imgOut[:self.y, :] = 0                      # set pixels above ROI to zero
        imgOut[self.y + self.height:, :] = 0        # set pixels below ROI to zero
        imgOut[:, :self.x] = 0                      # set pixels to the left of ROI to zero
        imgOut[:, self.x + self.width:] = 0         # set pixels to the right of ROI to zero
        
        return imgOut

    def clear_inside(self, img):
        """Set pixels in img to be zero if inside ROI"""
        img[self.y:self.y + self.height, self.x:self.x + self.width] = 0

        return img