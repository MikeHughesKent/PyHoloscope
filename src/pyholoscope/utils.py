# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

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
    """Returns 8 bit representation of amplitude and phase of field.

    Returns a tuple of amplitude and phase, both real 2D numpy arrays of type
    uint8. Amplitude is scaled between 0 and 255, phase is wrapped and mapped 
    to between 0 and 255, with 0 = 0 radians and 255 = 2pi radians.

    Parameters:
          img   : numpy.ndarray
                  2D numpy array, complex or real

    Returns:
          tuple of (ndarray, ndarray) : 8 bit amplitude and phase maps
    """

    if np.iscomplexobj(img):
        amp = np.abs(img).astype("float")
    else:
        amp = img
    amp = amp - np.min(amp)

    if np.max(amp) != 0:
        amp = amp / np.max(amp) * 255

    amp = amp.astype("uint8")

    phase = np.angle(img).astype("float")
    phase = phase % (2 * math.pi)
    phase = phase / (2 * math.pi) * 255
    phase = phase.astype("uint8")

    return amp, phase


def get16bit(img):
    """Returns 16 bit representation of amplitude and phase of field.

    Returns a tuple of amplitude and phase, both real 2D numpy arrays of type
    uint16. Amplitude is scaled between 0 and 2^16 -1, phase is mapped to between
    0 and 2^16 - 1, with 0 = 0 radians and 2^16 - 1 = 2pi radians.

    Parameters:
          img   : numpy.ndarray
                  2D numpy array, complex or real

    Returns:
          tuple of (ndarray, ndarray) : 16 bit amplitude and phase maps


    """
    amp = np.abs(img).astype("double")
    amp = amp - np.min(amp)
    if np.max(amp) != 0:
        amp = amp / np.max(amp) * 65535
    amp = amp.astype("uint16")

    phase = np.angle(img).astype("double")
    phase = phase % (2 * math.pi)
    phase = phase / (2 * math.pi) * 65535
    phase = phase.astype("uint16")

    return amp, phase


def save_phase_image(img, filename):
    """Saves phase as 16 bit tif. The phase is scaled so that 2pi = 65536.

    Parameters:
          img      : numpy.ndarray
                     2D numpy array, either complex field or real (phase map)
          filename : str
                     path to file to save to. If exists will be over-written.
    """

    if np.iscomplexobj(img):
        phase = np.angle(img).astype("double")
    else:
        phase = img.astype("double")
    phase = phase % (2 * math.pi)
    phase = ((phase / (2 * math.pi)) * 65536).astype("uint16")

    im = Image.fromarray(phase)
    im.save(filename)


def magnitude(img):
    """Returns magnitude of complex image.

    Parameters:
        img        : numpy.ndarray
                     complex image

    Returns:
        numpy.ndarray  : magnitude image
    """
    return np.abs(img) ** 2


def amplitude(img):
    """Returns amplitude of complex image. Deprecated, use `amp` instead.

    Parameters:
        img        : numpy.ndarray
                     complex image

    Returns:
         numpy.ndarray : amplitude image
    """
    return np.abs(img)


def amp(img):
    """Returns amplitude of complex image.

    Parameters:
        img        : numpy.ndarray
                     complex image

    Returns:
         numpy.ndarray : amplitude image
    """
    return np.abs(img)

def phase(img):
    """Returns phase of complex image, between 0 and 2pi.

    Parameters:
        img        : numpy.ndarray
                     complex image

    Returns:
         numpy.ndarray : phase map
    """
    return np.angle(img) % (2 * math.pi)


def load_image(filename):
    """
    Loads an image or stack of images from a file. Supports all formats supported by PIL.

    Parameters:
        filename    : str or Path
                      path to file

    Returns:
        numpy.ndarray : 2D, 3D or 4D numpy array representing image or stack

    """
    im = Image.open(filename)

    if im.n_frames > 1:
        h, w = np.shape(im)
        dt = np.array(im).dtype
        stack = np.zeros((im.n_frames, h, w), dtype=dt)
        for i in range(im.n_frames):
            im.seek(i)
            stack[i, :, :] = np.array(im)
        return stack
    else:
        return np.array(Image.open(filename))


def save_image(img, file, autoscale=True):
    """Saves an image stored as numpy array to an 8 bit image file.

    Parameters:
        img    : numpy.ndarray
                 image to save
        file   : str
                 filename to save to, type will be determined by extension.

    Optional Keyword Arguments:
        autoscale : boolean
                    if True (default), image is scaled to use full bit depth

    """
    if autoscale:
        img = Image.fromarray(get8bit(img)[0])
    else:
        img = Image.fromarray(img.astype("uint8"))
    img.save(file)


def save_image16(img, file, autoscale=True):
    """Saves an image stored as numpy array to an 16 bit image file.

    Parameters:
        img    : numpy.ndarray
                 image to save
        file   : str
                 filename to save to including extension, type will be
                 determined by extension and must support 16 bit images (e.g. tif)

    Optional Keyword Arguments:
        autoscale : boolean
                    if True (default), image is scaled to use full bit depth


    """
    if autoscale:
        img = Image.fromarray(get16bit(img)[0])
    else:
        img = Image.fromarray(img.astype("uint16"))
    img.save(file)


def save_amplitude_image8(img, filename):
    """Saves amplitude information as an 8 bit tif.

    Parameters:
        img    : numpy.ndarray
                 image to save
        file   : str
                 filename to load image from, including extension.
    """

    im = Image.fromarray(get8bit(img)[0])
    im.save(filename)


def save_amplitude_image16(img, filename):
    """Saves amplitude information as a 16 bit tif.

    Parameters:
        img    : numpy.ndarray
                 image to save
        file   : str
                 filename to load image from, including extension.
    """

    amp = amplitude(img)

    im = Image.fromarray(get16bit(img)[0])
    im.save(filename)


def extract_central(img, boxSize=None):
    """Extracts square of size boxSize*2 from centre of img. If boxSize is
    not specified, the largest possible square will be extracted.

    Parameters:
        img        : numpy.ndarray
                     complex or real image

    Keyword Arguments:
        boxSize    : int or None
                     size of square to be extracted

    Returns:
        numpy.ndarray : central square from image

    """
    w = np.shape(img)[0]
    h = np.shape(img)[1]

    cx = w / 2
    cy = h / 2
    if boxSize is not None:
        boxSemiSize = min(cx, cy, boxSize)
    else:
        boxSemiSize = min(cx, cy)

    imgOut = img[
        math.floor(cx - boxSemiSize) : math.floor(cx + boxSemiSize),
        math.ceil(cy - boxSemiSize) : math.ceil(cy + boxSemiSize),
    ]

    return imgOut


def invert(img):
    """Inverts an image, largest value becomes smallest and vice versa.

    Parameters:
        img        : numpy.ndarray
                     numpy array, input image

    Returns:
        numpy.ndarray : inverted image
    """

    return np.max(img) - img


def dimensions(inp):
    """Helper to obtain width and height in functions which accept multiple
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
