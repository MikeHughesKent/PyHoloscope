# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

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
        self.x = max(int(x), 0)
        self.y = max(int(y), 0)
        self.width = max(int(width), 0)
        self.height = max(int(height), 0)

    def __str__(self):
        return (
            str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self.width)
            + ","
            + str(self.height)
        )

    def constrain(self, min_x, min_y, max_x, max_y):
        """Stops ROI exceeding a specified size by adjusting ROI coordinates and size.

        Arguments:
            min_x    : int
                       minimum x coordinate
            min_y    : int
                       minimum y coordinate
            max_x    : int
                       maximum x coordinate
            max_y    : int
                       maximum y coordinate
        """

        self.x = max(self.x, min_x, 0)
        self.y = max(self.y, min_y, 0)

        self.width = max(min(self.width, max_x - self.x), 0)
        self.height = max(min(self.height, max_y - self.y), 0)

    def crop(self, img):
        """Crop and image using the ROI.

        Arguments:
            img : numpy.ndarray
                  input image

        Returns:
            numpy.ndarray : cropped image
        """
        return img[self.y : self.y + self.height, self.x : self.x + self.width]

    def clear_outside(self, img):
        """Set pixels in img to be zero if outside ROI.

        Arguments:
            img : numpy.ndarray
                  input image

        Returns:
            numpy.ndarray : image with pixels outside ROI set to zero

        """
        imgOut = img.copy()
        imgOut[: self.y, :] = 0  # set pixels above ROI to zero
        imgOut[self.y + self.height :, :] = 0  # set pixels below ROI to zero
        imgOut[:, : self.x] = 0  # set pixels to the left of ROI to zero
        imgOut[:, self.x + self.width :] = 0  # set pixels to the right of ROI to zero

        return imgOut

    def clear_inside(self, img):
        """Set pixels in img to be zero if inside ROI.

        Arguments:
            img : numpy.ndarray
                  input image

        Returns:
            numpy.ndarray : image with pixels inside ROI set to zero
        """

        img[self.y : self.y + self.height, self.x : self.x + self.width] = 0

        return img
