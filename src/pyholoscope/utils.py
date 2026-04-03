# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

This file contains utility functions.

"""

import math
import numpy as np
import matplotlib.pyplot as plt 

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

    num_frames = getattr(im, "n_frames", 1)

    if num_frames > 1:
        example_img = np.array(im)
        h, w = np.shape(example_img)[0:2]
        dt = example_img.dtype
        stack = np.zeros((num_frames, h, w), dtype=dt)
        for i in range(num_frames):
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
        tuple of (int, int), height and width
    """

    if type(inp) is np.ndarray:
        h, w = np.shape(inp)[0:2]
    elif type(inp) is tuple:
        h, w = inp
    else:
        w = inp
        h = inp

    return int(h), int(w)


def cshow(img, dpi=100, figsize=(10, 5), phase_cmap="twilight", amp_cmap="gray", title=None):
    """ Displays a complex image as two subplots, one for amplitude and one for phase.

    Parameters:
        img        : numpy.ndarray
                     complex image to display
    Keyword Arguments:
        dpi        : int
                     resolution of figure in dots per inch (default = 100)
        figsize    : tuple of (float, float)
                     size of figure in inches (default = (10, 5))
        phase_cmap : str
                     name of matplotlib colormap to use for phase (default = 'twilight')
        amp_cmap   : str
                     name of matplotlib colormap to use for amplitude (default = 'gray')
        title      : str or None
                     title to set for whole figure (default = None)
    
    Returns:
        tuple of (Figure, (Axes, Axes)) : figure and axes objects from matplotlib
     """
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi = dpi)
    
    # Set title of whole figure
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    amp = ax1.imshow(amplitude(img), cmap=amp_cmap)
    ax1.set_title("Amplitude")
    ph = ax2.imshow(phase(img), cmap=phase_cmap, vmin=0, vmax=2 * np.pi)
    ax2.set_title("Phase")

    # Create a colourbar for the phase plot, scaled to show 0 to 2pi radians
    cbar_phase = plt.colorbar(ph, ax=ax2, fraction=0.046, pad=0.04)
    cbar_phase.set_label('Phase (radians)', rotation=270, labelpad=15)
    cbar_phase.set_ticks([0, np.pi, 2 * np.pi]) 
    cbar_phase.set_ticklabels(['0', 'π', '2π']) 
    
    # Create a colourbar for the amplitude plot
    cbar_amp = plt.colorbar(amp, ax=ax1, fraction=0.046, pad=0.04)
    cbar_amp.set_label('Amplitude', rotation=270, labelpad=15)

    # Increase spacing between subplots to prevent overlap of colourbars and titles
    plt.subplots_adjust(wspace=0.3, top=0.85)
    plt.show()

    return fig, (ax1, ax2)

    
