# -*- coding: utf-8 -*-
"""
PyHoloscope
Python package for holgoraphic microscopy

Roi: Class for region of interest.


@author: Mike Hughes
Applied Optics Group, Physics & Astronomy, University of Kent
"""


class roi:
    
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        
    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.width) + ',' + str(self.height)
    
    def constrain(self, minX, minY, maxX, maxY):
        """ Stop ROI exceeding image size"""
        self.x = max(self.x, minX)
        self.y = max(self.y, minY)

        self.width = min(self.width, maxX - minX + 1) 
        self.height = min(self.width, maxY - minY + 1)
        
    def crop (self, img):
        """ img is cropped to ROI  """  
        return img[self.x: self.x + self.width, self.y:self.y + self.height]
    