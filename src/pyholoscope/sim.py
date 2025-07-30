# -*- coding: utf-8 -*-
"""
PyHoloscope - Python package for holgoraphic microscopy

The Sim package provide utilites for simulating off-axis holograms.

"""

import math

import numpy as np

from pyholoscope import dimensions


def off_axis(
    object_field, wavelength, pixel_size, tilt_angle, rotation=math.pi / 4, OPD=0
):
    """Generates simulated off-axis hologram.

    Arguments:
        object_field : ndarray, complex
                      complex field representing object
        wavelength  : float
                      wavelength of simulated light source
        pixel_size   : float
                      real size of pixels in hologram
        tilt_angle   : float
                      angle of reference beam on camera in radians

    Keyword Arguments:
        rotation    : float
                      rotation of the tilt with respect to x axis in rad
                      (Default is pi/4 = 45 deg)
        OPD         : optical path difference between beams (Default is 0)

    """

    # Convert wavelength to wavenumber
    k = 2 * math.pi / wavelength

    # Check size of object
    num_points_y, num_points_x = np.shape(object_field)

    # Generate tilted reference field
    ref_field = np.ones((num_points_y, num_points_x)) + 1j * np.ones((num_points_y, num_points_x))
    (xM, yM) = np.meshgrid(range(num_points_x), range(num_points_y))

    # Effect of tilt
    ref_freq = k * math.sin(tilt_angle)
    ref_field = ref_field * np.exp(
        1j * (ref_freq * pixel_size * (xM * np.cos(rotation) + yM * np.sin(rotation)))
    )

    # Effect of OPD
    ref_field = ref_field * np.exp(1j * 2 * math.pi * OPD / wavelength)

    # Field at camera is superposition of images of object and reference.
    # We are assuming 1:1 magnification here.
    cam_field = ref_field + object_field

    # Intensity at camera is square of field,
    cam_intensity = np.abs(cam_field) ** 2

    return cam_intensity
