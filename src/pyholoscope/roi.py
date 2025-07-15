# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

Roi: Class for region of interest.
"""

class Roi:
    
    def __init__(self, x, y, width, height):
        """Initialise ROI.

        Arguments:
            x, y    : int
                      x and y coordinates of top-left corner
            width   : int
                      width of ROI
            height  : int
                      height of ROI
        """
        self.x = max(int(x),0)
        self.y = max(int(y),0)
        self.width = max(int(width),0)
        self.height = max(int(height),0)
        
    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.width) + ',' + str(self.height)
    
    def constrain(self, minX, minY, maxX, maxY):
        """ Stops ROI exceeding image size by adjusting ROI coordinates and size.

        Arguments:
            minX, minY  : minimum x and y values
            maxX, maxY  : maximum x and y values
        """

        self.x = max(self.x, minX, 0)
        self.y = max(self.y, minY, 0)

        self.width = max(min(self.width, maxX - self.x), 0)
        self.height = max(min(self.height, maxY - self.y), 0)
        

    def crop (self, img):
        """ Crop and image using the ROI.

        Arguments:
            img : numpy.ndarray
                  input image

        Returns:
            numpy.ndarray : cropped image
        """
        return img[self.y: self.y + self.height, self.x:self.x + self.width]
    
    
    def clear_outside(self, img):
        """Set pixels in img to be zero if outside ROI.
        
        Arguments:
            img : numpy.ndarray
                  input image

        Returns:
            numpy.ndarray : image with pixels outside ROI set to zero
        
        """
        imgOut = img.copy()
        imgOut[:self.y, :] = 0                      # set pixels above ROI to zero
        imgOut[self.y + self.height:, :] = 0        # set pixels below ROI to zero
        imgOut[:, :self.x] = 0                      # set pixels to the left of ROI to zero
        imgOut[:, self.x + self.width:] = 0         # set pixels to the right of ROI to zero
        
        return imgOut
    

    def clear_inside(self, img):
        """Set pixels in img to be zero if inside ROI.

        Arguments:
            img : numpy.ndarray
                  input image

        Returns:
            numpy.ndarray : image with pixels inside ROI set to zero
        """

        img[self.y:self.y + self.height, self.x:self.x + self.width] = 0

        return img