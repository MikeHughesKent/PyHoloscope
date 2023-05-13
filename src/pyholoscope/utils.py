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
    """ Returns 8 bit repr. of amplitude and phase of field. 

    Returns a tuple of amplitude and phase, both real 2D numpy arrays of type
    uint8. Amplitude is scaled between 0 and 255, phase is mapped to between
    0 and 255, with 0 = 0 radians and 255 = 2pi radians.
    
    Arguments:
          img   : 2D numpy array, complex
       
    """
    
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
  
    if np.max(amp) != 0:
        amp = amp / np.max(amp) * 255
    
    amp = amp.astype('uint8')
        
    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2* math.pi) * 255
    phase = phase.astype('uint8')
        
    return amp, phase


def get16bit(img):
    """ Returns 16 bit repr. of amplitude and phase of field. 

    Returns a tuple of amplitude and phase, both real 2D numpy arrays of type
    uint16. Amplitude is scaled between 0 and 2^16 -1, phase is mapped to between
    0 and 2^16 - 1, with 0 = 0 radians and 2^16 - 1 = 2pi radians.
    
    Arguments:
          img   : 2D numpy array, complex          
    """
    amp = np.abs(img).astype('double')
    amp = amp - np.min(amp)
    if np.max(amp) != 0:
        amp = amp / np.max(amp) * 65535
    amp = amp.astype('uint16')
    
    phase = np.angle(img).astype('double')
    phase = phase % (2 * math.pi)
    phase = phase / (2* math.pi) * 65535
    phase = phase.astype('uint16')
    
    return amp, phase


def save_phase_image(img, filename):
    """ Saves phase as 16 bit tif. 
    
    The phase is scaled so that 2pi = 65536.
    
    Arguments:
          img      : 2D numpy array, either complex field or real (phase map)
          filename : str, file to save to. If exists will be over-written.
    """
    
    if np.iscomplexobj(img):
        phase = np.angle(img).astype('double')
    else:
        phase = img.astype('double')
    phase = phase - np.min(phase)    
    phase = ((phase / (2 * math.pi)) * 65536).astype('uint16')

    im = Image.fromarray(phase)
    im.save(filename)
        
    
def amplitude(img):
    """ Returns amplitude of complex image
    """
    return np.abs(img)
       
         
def phase(img):
    """ Returns phase of complex image, between 0 and 2pi.
    """ 
    return np.angle(img) % (2 * math.pi)  
        
       
def extract_central(img, boxSize):
       """ Extracts sqaure of size boxSize from centre of img
       """
       w = np.shape(img)[0]
       h = np.shape(img)[1]

       cx = w/2
       cy = h/2
       boxSemiSize = min(cx,cy,boxSize)
        
       img = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
       return img
       
    
def circ_cosine_window(imgSize, circleRadius, skinThickness, dataType = 'float32'):
    """ Produce a circular cosine window mask on grid of imgSize * imgSize. 
    
    Mask
    is 0 for radius > circleSize and 1 for radius < (circleSize - skinThickness)
    The intermediate region is a smooth cosine function.
    """
    if type(imgSize) is np.ndarray:
        (h,w) = np.shape(imgSize)[:1]       
    elif type(imgSize) is tuple:
        w,h = imgSize
    else:
        w = imgSize
        h = imgSize
        
    innerRad = circleRadius - skinThickness
    yM, xM = np.meshgrid(range(h),range(w))
    imgRad = np.sqrt( (yM - h/2) **2 + (xM - w/2) **2)
    mask =  np.cos(math.pi / (2 * skinThickness) * (imgRad - innerRad))**2
    mask[imgRad < innerRad ] = 1
    mask[imgRad > innerRad + skinThickness] = 0

    return mask.astype(dataType)



def square_cosine_window(imgSize, radius, skinThickness, dataType = 'float32'):
    """ Produce a square cosine window mask on grid of imgSize * imgSize. 
    
    Mask is 0 for radius > circleSize and 1 for radius < (circleSize - 
                                                          skinThickness)
    The intermediate region is a smooth cosine function.
    """   
     
    if type(imgSize) is np.ndarray:
        (h,w) = np.shape(imgSize)[:2]       
    elif type(imgSize) is tuple:
        w,h = imgSize
    else:
        w = imgSize
        h = imgSize

    innerRad = radius - skinThickness

    xCentre = int(w/2)
    yCentre = int(h/2)

    yR = np.arange(h)
    xR = np.arange(w)

    row = np.transpose(np.atleast_2d(np.cos(math.pi / (2 * skinThickness) * (xR - w / 2 - innerRad))**2))
    row[np.abs(xR - xCentre) < innerRad] = 1
    row[np.abs(xR - xCentre) > innerRad + skinThickness] = 0

    col = np.cos(math.pi / (2 * skinThickness) * (yR - h / 2 - innerRad))**2
    col[np.abs(yR - yCentre) < innerRad] = 1
    col[np.abs(yR - yCentre) > innerRad + skinThickness] = 0


    maskH = np.tile(col, (w, 1))
    maskV = np.tile(row, (1,h))

    mask = maskH * maskV
    
    
    return mask.astype(dataType)


def pil2np(im):
    """ Utility to convert PIL image 'im' to numpy array. Deprecated."""
    return  np.array(im)


def load_image(file):
    """ Loads an image from a file and returns as numpy array. 
    
    If the image is not square then the largest possible central sqaure is 
    extracted and returned.
    """
    img = Image.open(file)
    im = pil2np(img)
    
    # Make sure image is square
    if np.shape(im)[0] != np.shape(im)[1]:
        im = extract_central(im, min(np.shape(im)))
    return im


def save_image(file, img):
    """ Saves an image stored as numpy array to an 8 bit tif.
    """
    img = Image.fromarray(get8bit(img)[0])       
    img.save(file)
    
    
def save_amplitude_image8(img, filename):
    """ Saves amplitude information as an 8 bit tif"""    
    
    im = Image.fromarray(get8bit(img)[0])
    im.save(filename)
    
    
def save_amplitude_image16(img, filename):
    """ Saves amplitude information as 16 bit, normalised"""        
    amp = amplitude(img)
     
    im = Image.fromarray(get16bit(img)[0])
    im.save(filename)
    
    
def extract_central(img, boxSize):
    """ Extract a central square from an image. 
    
    The extracted square is centred on the input image, with size 2 * boxSize 
    if possible, otherwise the largest square that can be extracted.
    
    Arguments:
        img        : 2D numpy array, input image
        boxSize    : int, half of side length
     
    """
    w = np.shape(img)[0]
    h = np.shape(img)[1]

    cx = w/2
    cy = h/2
    boxSemiSize = min(cx,cy,boxSize)
    
    imgOut = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
    
    return imgOut