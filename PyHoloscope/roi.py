# -*- coding: utf-8 -*-
"""
Class for region of interest.


@author: Mike Hughes
Applied Optics Group, Physics & Astronomy, University of Kent
"""


class roi:
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.width) + ',' + str(self.height)
    
    # Stops ROI exceeding images size
    def constrain(self, minX, minY, maxX, maxY):
        self.x = max(self.x, minX)
        self.y = max(self.y, minY)

        self.width = min(self.width, maxX - minX + 1) 
        self.height = min(self.width, maxY - minY + 1)
        
    # img is cropped to ROI    
    def crop (self, img):
        return img[self.x: self.x + self.width, self.y:self.y + self.height]
    