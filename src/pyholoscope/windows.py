"""
PyHoloscope - Fast Holographic Microscopy for Python

This file contains functions for generating windows
for use in holographic microscopy.

"""

import numpy as np
import math
from pyholoscope import dimensions


def circ_window(img_size, circle_radius, data_type="float32"):
    """Produces a circular or elipitcal mask on grid of img_size as a numpy array.

    Parameters:
        img_size      : int or (int, int)
                       size of output array. Provide a single int to generate a square
                       image of that size, otherwise provide (w,h) to produce a rectangular
                       image, or 2D numpy array in which case the size of the array
                       will be used.
        circle_radius : float or (float, float)
                       Pixel values inside this radius will be 1. Provide a tuple
                       of (x,y) to have different x and y radii.

    Keyword Arguments:
        data_type     : str
                       data type of returned array (default is 'float32')

    Returns:
        numpy.ndarray : 2D numpy array containing mask


    """

    circle_radius = dimensions(circle_radius)
    [xM, yM] = np.meshgrid(range(circle_radius[0] * 2), range(circle_radius[1] * 2))
    mask = ((yM - circle_radius[1]) / circle_radius[1]) ** 2 + (
        (xM - circle_radius[0]) / circle_radius[0]
    ) ** 2 < 1

    return mask.astype(data_type)


def circ_cosine_window(img_size, circle_radius, skin_thickness, data_type="float32"):
    """Produces a circular or elliptical cosine window mask on grid of img_size as a numpy array.

    Parameters:
        img_size      :  int or (int, int)
                        size of output array. Provide a single int to generate a square
                       array of that size, otherwise provide (w,h) to produce a rectangular
                       array, or 2D numpy array in which case the size of the array
                       will be used.
        circle_radius :  float or (float, float)
                        Pixel values inside this radius will be 1. Provide a tuple
                        of (x,y) to have different x and y radii.
        skin_thickness : float
                        size of smoothed area inside circle/ellipse

    Keyword Arguments:
        data_type      : str
                        data type of returned array (default is 'float32')

    Returns:
        numpy.ndarray : 2D numpy array containing mask


    """
    h, w = dimensions(img_size)

    circle_radius = dimensions(circle_radius)
    skin_thickness = dimensions(skin_thickness)

    xM, yM = np.meshgrid(range(w), range(h))
    xMc = xM - w / 2
    yMc = yM - h / 2

    dist = np.sqrt(xMc**2 + yMc**2)
    xMc[xMc == 0] = 0.001
    theta = np.arctan(yMc / xMc)

    a = circle_radius[1]
    b = circle_radius[0]
    a2 = circle_radius[1] - skin_thickness[1]
    b2 = circle_radius[0] - skin_thickness[0]

    # Equations for radius of an ellipse at a given theta
    outerRadius = a * b / np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)

    innerRadius = (
        a2 * b2 / np.sqrt((a2 * np.cos(theta)) ** 2 + (b2 * np.sin(theta)) ** 2)
    )

    weight = (dist - innerRadius) / np.sqrt(
        (np.cos(theta) * skin_thickness[0]) ** 2
        + (np.sin(theta) * skin_thickness[1]) ** 2
    )

    # Smooth part of mask
    mask = np.cos(math.pi / 2 * (weight)) ** 2

    mask[weight > 1] = 0
    mask[weight < 0] = 1

    # Centre point will be NaN due to sqrt of 0
    mask[np.isnan(mask)] = 1

    return mask.astype(data_type)


def square_cosine_window(img_size, radius=None, skin_thickness=0, data_type="float32"):
    """Produces a square/rectangular cosine window mask on grid of img_size * img_size.
    Mask is 0 for pixels > radius and 1 for pixels < (radius -
    skin_thickness).  The intermediate region is a smooth squared cosine function.

    Arguments:
        img_size       : int or (int, int)
                        size of output array. Provide a single int to generate a square
                       array of that size, otherwise provide (w,h) to produce a rectangular
                       array, or 2D numpy array in which case the size of the array
                       will be used.

    Keyword Arguments:
        radius        :  float or (float, float) or None
                        Pixel values inside this radius will be 1. Provide a tuple
                        of (x,y) to have different x and y radii. If None (default), the radius will match
                        the size of the image.
        skin_thickness : float or None
                        size of smoothed area inside circle/ellipse. Default is 0.
        data_type      : str
                        data type of returned array (default is 'float32')

    Returns:
        numpy.ndarray : 2D numpy array containing mask


    """

    h, w = dimensions(img_size)

    if radius is None:
        radiusX = w / 2
        radiusY = h / 2
    elif type(radius) is tuple:
        radiusX, radiusY = radius
    else:
        radiusX = radius
        radiusY = radius

    innerRadX = radiusX - skin_thickness
    innerRadY = radiusY - skin_thickness

    xCentre = int(w / 2)
    yCentre = int(h / 2)

    yR = np.arange(h)
    xR = np.arange(w)

    row = np.cos(math.pi / (2 * skin_thickness) * (np.abs(xR - w / 2) - innerRadX)) ** 2
    row[np.abs(xR - xCentre) < innerRadX] = 1
    row[np.abs(xR - xCentre) > innerRadX + skin_thickness] = 0

    col = np.transpose(
        np.atleast_2d(
            np.cos(math.pi / (2 * skin_thickness) * (np.abs(yR - h / 2) - innerRadY))
            ** 2
        )
    )
    col[np.abs(yR - yCentre) < innerRadY] = 1
    col[np.abs(yR - yCentre) > innerRadY + skin_thickness] = 0

    maskH = np.tile(col, (1, w))
    maskV = np.tile(row, (h, 1))

    mask = maskH * maskV

    return mask.astype(data_type)
