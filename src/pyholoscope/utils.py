# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holgoraphic microscopy

@author: Mike Hughes, Applied Optics Group, University of Kent

This file contains utility functions.


"""
import math
import numpy as np

try:
    import cupy as cp
except:
    pass

from PIL import Image


def get8bit(img):
    """ Returns 8 bit representation of amplitude and phase of field. 

    Returns a tuple of amplitude and phase, both real 2D numpy arrays of type
    uint8. Amplitude is scaled between 0 and 255, phase is mapped to between
    0 and 255, with 0 = 0 radians and 255 = 2pi radians.

    Parameters:
          img   : ndarray
                  2D numpy array, complex or real
              
    Returns:
          tuple of (ndarray, ndarray), 8 bit amplitude and phase maps
    """

    if np.iscomplexobj(img):
        amp = np.abs(img).astype('float')
    else:
        amp = img
    amp = amp - np.min(amp)

    if np.max(amp) != 0:
        amp = amp / np.max(amp) * 255

    amp = amp.astype('uint8')

    phase = np.angle(img).astype('float')
    phase = phase % (2 * math.pi)
    phase = phase / (2 * math.pi) * 255
    phase = phase.astype('uint8')

    return amp, phase


def get16bit(img):
    """ Returns 16 bit representation of amplitude and phase of field. 

    Returns a tuple of amplitude and phase, both real 2D numpy arrays of type
    uint16. Amplitude is scaled between 0 and 2^16 -1, phase is mapped to between
    0 and 2^16 - 1, with 0 = 0 radians and 2^16 - 1 = 2pi radians.

    Parameters:
          img   : ndarray
                  2D numpy array, complex or real
                  
    Returns:
          tuple of (ndarray, ndarray), 16 bit amplitude and phase maps

               
    """
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
    if np.max(amp) != 0:
        amp = amp / np.max(amp) * 65535
    amp = amp.astype('uint16')

    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2 * math.pi) * 65535
    phase = phase.astype('uint16')

    return amp, phase


def save_phase_image(img, filename):
    """ Saves phase as 16 bit tif. The phase is scaled so that 2pi = 65536.

    Parameters:
          img      : ndarray
                     2D numpy array, either complex field or real (phase map)
          filename : str
                     path to file to save to. If exists will be over-written.
    """

    if np.iscomplexobj(img):
        phase = np.angle(img).astype('double')
    else:
        phase = img.astype('double')
    phase = phase - np.min(phase)
    phase = ((phase / (2 * math.pi)) * 65536).astype('uint16')

    im = Image.fromarray(phase)
    im.save(filename)


def magnitude(img):
    """ Returns magnitude of complex image.
    
    Parameters:
        img        : ndarray
                     complex image
                     
    Returns:
         ndarray, magnitude image                 
    """
    return np.abs(img)**2


def amplitude(img):
    """ Returns amplitude of complex image.
    
    Parameters:
        img        : ndarray
                     complex image
                     
    Returns:
         ndarray, amplitude image  
    """
    return np.abs(img)


def phase(img):
    """ Returns phase of complex image, between 0 and 2pi.
    
    Parameters:
        img        : ndarray
                     complex image
                     
    Returns:
         ndarray, phase map                  
    """
    return np.angle(img) % (2 * math.pi)




def circ_window(imgSize, circleRadius, dataType = 'float32'):
    """ Produces a circular or elipitcal mask on grid of imgSize.

    Parameters:
        imgSize      : int or (int, int)
                       size of output image. Provide a single int to generate a square
                       image of that size, otherwise provide (w,h) to produce a rectangular
                       image.
        circleRadius : float or (float, float)
                       Pixel values inside this radius will be 1. Provide a tuple
                       of (x,y) to have different x and y radii.
      
    Keyword Arguments:
        dataType     : str
                       data type of returned array (default is 'float32')

    Returns:
        ndarray, 2D numpy array containing mask               


    """  
    
    circleRadius = dimensions(circleRadius)
    [xM, yM] = np.meshgrid(range(circleRadius[0] *2), range(circleRadius[1] * 2))
    mask = ( (yM - circleRadius[1]) / circleRadius[1] )**2 + ((xM - circleRadius[0]) / circleRadius[0])**2 < 1

    return mask.astype(dataType)


def circ_cosine_window(imgSize, circleRadius, skinThickness, dataType='float32'):
    """ Produces a circular or elliptical cosine window mask on grid of imgSize.

    Parameters:
        imgSize      :  int or (int, int)
                        size of output image. Provide a single int to generate a square
                        image of that size, otherwise provide (w,h) to produce a rectangular
                        image.
        circleRadius :  float or (float, float)
                        Pixel values inside this radius will be 1. Provide a tuple
                        of (x,y) to have different x and y radii.
        skinThickness : float
                        size of smoothed area inside circle/ellipse

    Keyword Arguments:
        dataType      : str
                        data type of returned array (default is 'float32')

    Returns:
        ndarray, 2D numpy array containing mask               


    """
    w, h = dimensions(imgSize)

    circleRadius = dimensions(circleRadius)
    skinThickness = dimensions(skinThickness)        

    xM, yM = np.meshgrid(range(w), range(h))
    xMc = (xM - w/2)
    yMc = (yM - h/2)

    dist = np.sqrt(xMc**2 + yMc**2)
    xMc[xMc == 0] = 0.001
    theta = np.arctan(yMc / xMc)

    a = circleRadius[1]
    b = circleRadius[0]
    a2 = circleRadius[1] - skinThickness[1]
    b2 = circleRadius[0] - skinThickness[0]

    # Equations for radius of an ellipse at a given theta
    outerRadius = a * b / \
        np.sqrt((a * np.cos(theta))**2 + (b * np.sin(theta))**2)

    innerRadius = a2 * b2 / \
        np.sqrt((a2 * np.cos(theta))**2 + (b2 * np.sin(theta))**2)

    weight = (dist - innerRadius) / np.sqrt( (np.cos(theta) * skinThickness[0])**2 + (np.sin(theta) * skinThickness[1])**2)
    
    # Smooth part of mask
    mask = np.cos(math.pi / 2 * (weight))**2
  
    mask[weight > 1] = 0
    mask[weight < 0] = 1
  
    # Centre point will be NaN due to sqrt of 0  
    mask[np.isnan(mask)] = 1

    return mask.astype(dataType)


def square_cosine_window(imgSize, radius, skinThickness, dataType='float32'):
    """ Produces a square cosine window mask on grid of imgSize * imgSize. 
    Mask is 0 for radius > circleSize and 1 for radius < (circleSize - 
    skinThickness).  The intermediate region is a smooth squared cosine function.
    
    Parameters:
        imgSize       : int or (int, int)
                        size of output image. Provide a single int to generate a square
                        image of that size, otherwise provide (w,h) to produce a rectangular
                        image.
        circleRadius :  float or (float, float)
                        Pixel values inside this radius will be 1. Provide a tuple
                        of (x,y) to have different x and y radii.
        skinThickness : float
                        size of smoothed area inside circle/ellipse

    Keyword Arguments:
        dataType      : str
                        data type of returned array (default is 'float32')

    Returns:
        ndarray, 2D numpy array containing mask   
    
    
    """

    w, h = dimensions(imgSize)

    if type(radius) is tuple:
        radiusX, radiusY = radius
    else:
        radiusX = radius
        radiusY = radius

    innerRadX = radiusX - skinThickness
    innerRadY = radiusY - skinThickness

    xCentre = int(w/2)
    yCentre = int(h/2)

    yR = np.arange(h)
    xR = np.arange(w)

    row = np.cos(math.pi / (2 * skinThickness) *
                 (np.abs(xR - w/2) - innerRadX))**2
    row[np.abs(xR - xCentre) < innerRadX] = 1
    row[np.abs(xR - xCentre) > innerRadX + skinThickness] = 0

    col = np.transpose(np.atleast_2d(
        np.cos(math.pi / (2 * skinThickness) * (np.abs(yR - h/2) - innerRadY))**2))
    col[np.abs(yR - yCentre) < innerRadY] = 1
    col[np.abs(yR - yCentre) > innerRadY + skinThickness] = 0

    maskH = np.tile(col, (1, w))
    maskV = np.tile(row, (h, 1))

    mask = maskH * maskV

    return mask.astype(dataType)


def pil2np(im):
    """ Utility to convert PIL image 'im' to numpy array. Deprecated."""
    return np.array(im)


def load_image(file, square=False):
    """ Loads an image from a file and returns as numpy array. 

    Parameters:
        file   : str
                 filename to load image from, including exension.

    Keyword Arguments:
        square : boolean
                 if True, non-square will be made square by taking the largest
                 possible central square, default is False.
                 
    Returns:
        ndarray, 2D image
    """
    img = Image.open(file)
    im = pil2np(img)

    if square:
        # Make sure image is square
        if np.shape(im)[0] != np.shape(im)[1]:
            im = extract_central(im, min(np.shape(im)[0:2]))

    return im


def save_image(img, file):
    """ Saves an image stored as numpy array to an 8 bit tif.
    
    Parameters:
        img    : ndarray
                 image to save
        file   : str
                 filename to load image from, including extension.
    
    """
    img = Image.fromarray(get8bit(img)[0])
    img.save(file)


def save_amplitude_image8(img, filename):
    """ Saves amplitude information as an 8 bit tif.
   
    Parameters:
        img    : ndarray
                 image to save
        file   : str
                 filename to load image from, including extension.
    """

    im = Image.fromarray(get8bit(img)[0])
    im.save(filename)


def save_amplitude_image16(img, filename):
    """ Saves amplitude information as a 16 bit tif.
    
    Parameters:
        img    : ndarray
                 image to save
        file   : str
                 filename to load image from, including extension.
    """
    
    amp = amplitude(img)

    im = Image.fromarray(get16bit(img)[0])
    im.save(filename)


def extract_central(img, boxSize=None):
    """ Extracts square of size boxSize*2 from centre of img. If boxSize is
    not specified, the largest possible square will be extracted.
    
    Parameters:
        img        : ndarray
                     complex or real image
                     
    Keyword Arguments:
        boxSize    : int or None
                     size of square to be extracted                 
                     
    Returns:
         ndarray, central square from image 
    
    """
    w = np.shape(img)[0]
    h = np.shape(img)[1]

    cx = w/2
    cy = h/2
    if boxSize is not None:
        boxSemiSize = min(cx, cy, boxSize)
    else:
        boxSemiSize = min(cx, cy)

    imgOut = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize),
                 math.ceil(cy - boxSemiSize): math.ceil(cy + boxSemiSize)]

    return imgOut


def invert(img):
    """ Inverts an image, largest value becomes smallest and vice versa.

    Parameters:
        img        : ndarray
                     numpy array, input image
  
    Returns:
        ndarray, inverted image
    """

    return np.max(img) - img


def dimensions(inp):
    """ Helper to obtain width and height in functions which accept multiple
    ways to send this information. The input may either be a single value,
    for a square image, a tuple of (h, w) or a 2D array.
    
    Parameters:
        inp        : int or (int, int) or ndarray
        
    Returns:
        tuple of (int, int), width and height
    """

    if type(inp) is np.ndarray:
        h, w = np.shape(inp)[0:2]
    elif type(inp) is tuple:
        w, h = inp
    else:
        w = inp
        h = inp

    return int(w), int(h)
