# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holographic microscopy

This file contains functions for working with off-axis holograms.
"""

import math
import numpy as np
import scipy

import matplotlib.pyplot as plt

try:
    import cupy as cp

    cuda_available = True
except:
    cuda_available = False

from pyholoscope.utils import extract_central, dimensions


def off_axis_demod(
    hologram,
    crop_centre,
    crop_radius,
    return_full=False,
    return_fft=False,
    mask=None,
    cuda=False,
):
    """Removes spatial modulation from off-axis hologram to obtain complex field.

    By default, returns the complex field as a 2D numpy array of size
    2 * crop_radius. If return_full is True, the returned
    array will instead by the same size as the input hologram. If return_fft is
    True, function returns a tuple (field, FFT) where FFT is a log scaled
    image of the FFT of the hologram (2D numpy array, real).

    Arguments:
          hologram   : numpy.ndarray
                       2D numpy array, real, raw hologram
          crop_centre : tuple of (int, int).
                       pixel location in FFT of modulation frequency
          crop_radius : int or (int, int)
                       semi-diameter of sqaure or rectangle to extract
                       around modulation frequency. Provide a single int
                       for a sqaure area or a tuple of (w,h) for a rectangle.

    Keyword Arguments:
          return_full : boolean
                       if True, the returned reconstruction will be the same
                       size as the input hologram, otherwise it will be
                       2 * crop_radius. (Default is False)
          return_fft  : boolean
                       if True will return a tuple of (demod image,
                       log scaled FFT) for display purposes
                       (Default is False)
          mask       : ndarray
                       2D complex array. Custom mask to use around
                       demodulation frequency. Must match size of
                       (crop_radius_x, crop_radius_y)
          cuda      :  boolean
                       if True GPU will be used if available.
    Returns:
          numpy. ndarray   : reconstructed field as complex numpy array or
                             tuple of (ndarray, ndarray) if return_fft is True
    """

    # Size of image in pixels
    height, width = np.shape(hologram)

    crop_centre = dimensions(crop_centre)
    crop_radius = dimensions(crop_radius)

    # Apply 2D FFT
    if cuda is False or cuda_available is False:
        camera_fft = scipy.fft.rfft2(hologram)
    else:
        if type(hologram) is np.ndarray:
            hologram = cp.array(hologram)
        if type(mask) is np.ndarray:
            mask = cp.array(mask)
        camera_fft = cp.fft.rfft2(cp.array(hologram))

    # Shift the ROI to the centre
    shifted_fft = camera_fft[
        round(crop_centre[1] - crop_radius[1]) : round(crop_centre[1] + crop_radius[1]),
        round(crop_centre[0] - crop_radius[0]) : round(crop_centre[0] + crop_radius[0]),
    ]

    # Apply the mask
    if mask is not None:
        assert np.shape(mask) == np.shape(shifted_fft), "Incorrect mask size."
        masked_fft = shifted_fft * mask
    else:
        masked_fft = shifted_fft

    if return_full:
        h, w = np.shape(hologram)
        (
            h2,
            w2,
        ) = np.shape(masked_fft)
        x, y = round((w - w2) / 2), round((h - h2) / 2)
        output = np.zeros((h, w), dtype="complex")
        output[y : y + h2, x : x + w2] = masked_fft
        masked_fft = output

    # Reconstruct complex field
    if cuda is False or cuda_available is False:
        recon_field = scipy.fft.ifft2(scipy.fft.fftshift(masked_fft))
    else:
        recon_field = cp.asnumpy(cp.fft.ifft2(cp.fft.fftshift(masked_fft)))

    if return_fft:
        if cuda is True and cuda_available is True:
            try:
                camera_fft = cp.asnumpy(camera_fft)
            except:
                pass
        return recon_field, np.log(np.abs(camera_fft) + 0.000001)  # Stops log(0)

    else:
        return recon_field


def off_axis_find_mod(hologram, mask_fraction=0.1):
    """Finds the location of the off-axis holography modulation peak in the FFT.

    Arguments:
          hologram     : ndarray
                         2D numpy array, real, raw hologram

    Keyword Arguments:
          mask_fraction : float
                         between 0 and 1, fraction of image around d.c. to
                         mask to avoid the d.c. peak being detected
                         (default = 0.1).
    Returns:
          tuple of (int, int), modulation location in FFT (y location, x location)
    """

    # Apply 2D FFT
    camera_fft = np.transpose(np.abs(scipy.fft.rfft2((hologram))))

    # Need to crop out DC otherwise we will find that. Set the areas around
    # dc (for both quadrants) to zero. The size of the masked area is mask_fraction * the
    # size of the image (smallest dimension)
    maskSize = int(np.min(np.shape(hologram)) * mask_fraction)

    camera_fft[:maskSize, :maskSize] = 0
    camera_fft[:maskSize:, -maskSize:] = 0

    peak_location = np.unravel_index(camera_fft.argmax(), camera_fft.shape)

    return peak_location


def off_axis_find_crop_radius(hologram, mask_fraction=0.1):
    """Estimates the off-axis crop radius based on modulation peak position. If the
    hologram is square, this is the radius of a circle, otherwise if it is rectangular
    than the crop radius is a tuple of (y radius, x radius), corresponding to
    half the lengths of the two axes of an ellipse.

    Arguments:
          hologram     : numpy.ndarray
                         raw hologram, 2D real array

    Keyword Arguments:
          mask_fraction : float
                         between 0 and 1, fraction of image around d.c. to
                         mask to avoid the d.c. peak being detected
                         (default = 0.1).
    Returns:
          tuple of (int, int) = (y radius, x radius)
    """

    h = np.shape(hologram)[0]
    w = np.shape(hologram)[1]

    peak_loc_y, peak_loc_x = off_axis_find_mod(hologram, mask_fraction=mask_fraction)

    # The crop radii will have the same ratio as the width and height of the hologram
    aspect_ratio = h / w

    peak_loc_square = (peak_loc_y * aspect_ratio, peak_loc_x)

    # In the optimal case, the radius is 1/3rd of the modulation position
    if peak_loc_x < h / 2:
        crop_radius_square = np.sqrt(peak_loc_square[0] ** 2 + peak_loc_square[1] ** 2) / 3
        crop_radius_square = min(
            crop_radius_square,
            peak_loc_y,
            int(w - peak_loc_y),
            peak_loc_x,
            int(h / 2 - peak_loc_x),
        )
    else:
        crop_radius_square = (
            np.sqrt(peak_loc_square[0] ** 2 + (w * 2 - peak_loc_square[1]) ** 2) / 3
        )
        crop_radius_square = min(
            crop_radius_square,
            peak_loc_y,
            int(w - peak_loc_y),
            peak_loc_x - h / 2 * aspect_ratio,
            int(h * aspect_ratio - peak_loc_x),
        )

    crop_radius_x = int(round(crop_radius_square))
    crop_radius_y = int(round(crop_radius_square / aspect_ratio))

    return crop_radius_y, crop_radius_x


def off_axis_predict_mod(wavelength, pixel_size, num_pixels, tilt_angle, rotation=0):
    """Predicts the location of the modulation peak in the FFT.

    Arguments:
          wavelegnth   : float
                         light wavelength in metres
          pixel_size    : float
                         hologram physical pixel size in metres
          num_pixels    : int or (int, int)
                         hologram size in pixels,
          tilt_angle    : float
                         angle of reference beam on camera in radians

    Keyword Arguments:
          rotation     : float
                         rotation of tilt with respect to x axis, in radians (default is 0)

    Returns:
          tuple of (int, int), location of modulation (x pixel, y pixel)

    """

    # Spatial frequency of modulation
    ref_freq = math.sin(tilt_angle) / wavelength

    # Spatial frequency at edge of FFT
    max_spatial_freq = 1 / (pixel_size * 2)

    im_size_x, im_size_y = dimensions(num_pixels)

    # Pixel corresponding to frequency in Fourier Domain
    if rotation % math.pi < math.pi / 2:
        mod_freq_px_x = round(ref_freq / max_spatial_freq * np.abs(np.cos(rotation)) * im_size_x / 2)
        mod_freq_px_y = round(ref_freq / max_spatial_freq * np.abs(np.sin(rotation)) * im_size_y / 2)
    else:
        mod_freq_px_x = round(
            ref_freq / max_spatial_freq * np.abs(np.cos(math.pi - rotation)) * im_size_x / 2
        )
        mod_freq_px_y = im_size_y - round(
            ref_freq / max_spatial_freq * np.abs(np.sin(rotation)) * im_size_y / 2
        )

    if mod_freq_px_x < 0:
        mod_freq_px_x = mod_freq_px_x + im_size_x
    if mod_freq_px_y < 0:
        mod_freq_px_y = mod_freq_px_y + im_size_y

    return mod_freq_px_x, mod_freq_px_y


def off_axis_predict_mod_distance(
    wavelength, pixel_size, num_pixels, tilt_angle, rotation=0
):
    """Predicts the absolute distance of the modulation peak in the FFT from the dc.

    Arguments:
          wavelegnth   : float
                         light wavelength in metres
          pixel_size    : float
                         hologram physical pixel size in metres
          num_pixels    : int or (int, int)
                         hologram size in pixels,
          tilt_angle    : float
                         angle of reference beam on camera in radians

    Keyword Arguments:
          rotation     : float
                         rotation of tilt with respect to x axis, in radians (default is 0)

    Returns:
          float        : distance in pixels

    """

    x, y = off_axis_predict_mod(wavelength, pixel_size, num_pixels, tilt_angle, rotation)

    return math.sqrt(x**2 + y**2)


def off_axis_predict_tilt_angle(hologram, wavelength, pixel_size, mask_fraction=0.1):
    """Returns the reference beam tilt based on the hologram modulation. The angle
    is returned in radians.

    Arguments:
          hologram     : ndarray
                         2D numpy array, real, hologram
          wavelength   : float
                         light wavelength in metres
          pixel_size    : float
                         hologram physical pixel size in metres

    Optional Keyword Arguments:
          mask_fraction : float
                         between 0 and 1, fraction of image around d.c. to
                         mask to avoid the d.c. peak being detected
                         (default = 0.1).

    Returns:
          float        : tilt angle in radians
    """

    # Wavenumber
    k = 2 * math.pi / wavelength

    h, w = np.shape(hologram)[:2]

    # Find the location of the peak
    peak_location = off_axis_find_mod(hologram, mask_fraction=mask_fraction)

    # Pixel sizes in FFT (the spatial frequency)
    v_pixel_spatial_freq = 1 / (pixel_size * np.shape(hologram)[0])
    h_pixel_spatial_freq = 1 / (pixel_size * np.shape(hologram)[1])

    # Depending on quadrant could be relative to either top-left or
    # top-right corner, so check both and use the closest distance
    peak_dist1 = np.sqrt((v_pixel_spatial_freq * peak_location[1]) ** 2 + (h_pixel_spatial_freq * peak_location[0]) ** 2)
    peak_dist2 = np.sqrt(
        (v_pixel_spatial_freq * peak_location[1]) ** 2 + (h_pixel_spatial_freq * (peak_location[0] - w)) ** 2
    )
    spatial_freq = min(peak_dist1, peak_dist2)

    tilt_angle = math.asin(2 * math.pi * spatial_freq / k)

    return tilt_angle
