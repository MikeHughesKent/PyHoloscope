# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

PropLUT is the class for generating and storing propagator look up tables.

"""

import numpy as np

from pyholoscope.focusing import propagator
from pyholoscope.focusing_numba import propagator_numba
from pyholoscope.utils import dimensions
from pyholoscope.propagator import Propagator


class PropLUT:
    """Stores a propagator look up table (LUT).

    The LUT contains angular spectrum propagators for the specified parameters.

    """

    def __init__(
        self,
        img_size,
        wavelength,
        pixel_size,
        depth_range,
        num_depths,
        geometry="plane",
        use_numba=True,
        precision="single",
    ):
        """Creates a propagator look up table (LUT) containing angular spectrum propagators.

        Arguments:
            img_size   : int or tuple of (int, int)
                        size of propagators, square or rectangular
            wavelength: float
                        wavelength of light
            pixel_size : float
                        physical size of pixels
            depth_range: tuple
                        range of depths to generate propagators for
            num_depths : int
                        number of depths to generate propagators for

        Keyword Arguments:
            geometry  : str
                        'plane' (default) or 'point'
            numba     : bool
                        flag to use numba for speed up (default = False)
            precision : str
                        'single' or 'double' (default = 'single')
        """

        if precision == "double":
            dataType = "complex128"
        else:
            dataType = "complex64"

        self.depths = np.linspace(depth_range[0], depth_range[1], num_depths)
        self.size = img_size
        self.num_depths = num_depths
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.geometry = geometry
        h, w = dimensions(img_size)
        self.prop_table = np.zeros((num_depths, h, w), dtype=dataType)
        for idx, depth in enumerate(self.depths):
            self.prop_table[idx, :, :] = propagator(
                (h, w),
                wavelength,
                pixel_size,
                depth,
                geometry=geometry,
                use_numba=use_numba,
            ).propagator

    def propagator(self, depth):
        """Returns the propagator from the LUT which is closest to requested
        depth. If depth is outside the range of the propagators, function returns None.

        Parameters:
            depth     : float
                        refocus depth for requested propagator
        """

        # Find nearest propagator
        idx = self.closest_index(depth)
        if idx is not None:
            prop = Propagator(
                self.prop_table[idx, :, :],
                wavelength=self.wavelength,
                pixel_size=self.pixel_size,
                depth=self.depths[idx],
                geometry=self.geometry,
            )
            return prop
        else:
            return None

    """Returns the index of the propagator that is closest to requested
    depth. If depth is outside the range of the propagators, function returns None.

    Parameters:
        depth     : float
                    refocus depth for requested propagator
    """

    def closest_index(self, depth):
        if depth < self.depths[0] or depth > self.depths[-1]:
            return None

        elif self.num_depths == 1:  # Otherwise the algorithm to get the index will fail
            idx = 0

        else:
            idx = round(
                (depth - self.depths[0])
                / (self.depths[-1] - self.depths[0])
                * (self.num_depths - 1)
            )

        return idx

    def __str__(self):
        return (
            "LUT of "
            + str(self.num_depths)
            + " propagators from depth of "
            + str(self.depths[0])
            + " to "
            + str(self.depths[-1])
            + ". Wavelength: "
            + str(self.wavelength)
            + ", Pixel Size: "
            + str(self.pixel_size)
            + " ,Size:"
            + str(self.size)
        )
