# -*- coding: utf-8 -*-
"""
Focus scoring functions and registry.
"""

import numpy as np
import scipy
import cv2 as cv

def get_focus_score_methods():
    """Returns a list of available focus scoring method names."""

    seen = set()
    result = {}

    # For backwards compatibility, there are some keys in the methods dict that point to the same function.
    # The deprecated names are always defined second, so we can loop through and only 
    # return the first occurance of each value.
    for key, value in methods.items():
        if value not in seen:
            result[key] = value
            seen.add(value)

    return list(result.keys())

def brenner(img):
    """Brenner focus metric (lower is better)."""
    (h, w) = np.shape(img)
    BrennX = np.zeros((h, w))
    BrennY = np.zeros((h, w))
    BrennX[0:-2, :] = img[2:, :] - img[0:-2,]
    BrennY[:, 0:-2] = img[:, 2:] - img[:, 0:-2]
    scoreMap = np.maximum(BrennY**2, BrennX**2)
    return -np.mean(scoreMap)

def peak(img):
    """Peak intensity focus metric (lower is better)."""
    return -np.max(img)

def sobel(img):
    """Sobel gradient energy focus metric (lower is better)."""
    filtX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filtY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    xSobel = scipy.signal.convolve2d(img, filtX)
    ySobel = scipy.signal.convolve2d(img, filtY)
    sobel = xSobel**2 + ySobel**2
    return -np.mean(sobel)

def sobel_variance(img):
    """Variance of Sobel energy focus metric (lower is better)."""
    filtX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filtY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    xSobel = scipy.signal.convolve2d(img, filtX)
    ySobel = scipy.signal.convolve2d(img, filtY)
    sobel = xSobel**2 + ySobel**2
    return -(np.std(sobel) ** 2)

def variance(img):
    """Intensity standard deviation focus metric (lower is better)."""
    return -np.std(img)

# https://doi.org/10.1016/j.optlaseng.2020.106195
def dark_focus(img):
    """DarkFocus metric from Optics & Laser Eng. 2020 (lower is better)."""
    kernelX = np.array([[-1, 0, 1]])
    kernelY = kernelX.transpose()
    gradX = cv.filter2D(img, -1, kernelX)
    gradY = cv.filter2D(img, -1, kernelY)
    mean, stDev = cv.meanStdDev(gradX**2 + gradY**2)
    return -(stDev[0, 0] ** 2)

def norm_var(img):
    """ Normalised variance focus metric (lower is better). """
    return -np.sum((img - np.mean(img))**2) / np.mean(img)
    
def sum_focus(img):
    """Sum of pixel intensities (lower is better)."""
    return np.sum(img)    

methods= {
    'brenner': brenner,
    'sobel': sobel,
    'dark_focus': dark_focus,
    'DarkFocus': dark_focus,           # Backwards compatibility
    'sobel_variance': sobel_variance,
    'SobelVariance': sobel_variance,   # Backwards compatibility
    'peak': peak,
    'norm_var': norm_var,
    'sum': sum_focus,
}

def focus_score(img, method):
    """Returns score of how 'in focus' an amplitude image is.
    Score is returned as a float, the lower the better the focus.

    Arguments:
        img          : numpy.ndarray
                       image to score, 2D real array
        method       : str or callable
                       scoring method name, or a callable accepting a 2D array
                       and returning a float (lower is better). Built-ins are:
                       'Brenner', 'Sobel', 'SobelVariance', 'Var',
                       'DarkFocus' or 'Peak'.
    Returns:
        float        : focus score
    """
    if isinstance(method, str):
        method = method.lower()
        if method not in methods:
            raise ValueError(f"Unknown focus scoring method: {method}")
        method = methods[method]
    elif not callable(method):
        raise ValueError("Method must be a string or a callable function.")   

    return  sum_focus(img)

