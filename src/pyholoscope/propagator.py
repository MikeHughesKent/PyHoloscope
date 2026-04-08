"""
PyHoloscope - Fast Holographic Microscopy for Python

The Propagator class is a container to hold a propagator for wavefront
propagation in holography along with metadata. It does not generate the
propagator.
"""

import numpy as np


class Propagator:
    """Represents a propagator for wavefront propagation in holography.

    Attributes:

        propagator  : 2D complex numpy array containing the propagator.

        wavelength  : The wavelength of the light used in the propagation (m).

        pixel_size  : The true physical size of the pixels in the hologram (m).

        magnified_pixel_size : The effective pixel size used to generate the
                       propagator after any magnification correction (m).

        depth       : True propagation distance (m).

        magnified_depth : Effective propagation distance used to generate the
                  propagator after any magnification correction (m).
    """

    wavelength = None
    pixel_size = None
    magnified_pixel_size = None
    depth = None
    magnified_depth = None
    propagator = None
    shape = None
    propagation_method = None
    correct_pixel_size = None
    source_distance = None

    def __init__(
        self,
        propagator,
        wavelength=None,
        pixel_size=None,
        magnified_pixel_size=None,
        depth=None,
        magnified_depth=None,
        propagation_method=None,
        correct_pixel_size=None,
        source_distance=None,
    ):
        """
        Initializes the Propagator instance by stoing the propagator and its attributes.

        Parameters:
            propagator: 2D complex numpy array containing the propagator.

        Optional Keyword Arguments:
            wavelength: float
                        The wavelength of the light (m).
            pixel_size: float
                       The true size of the pixels in the hologram (m).
                 magnified_pixel_size: float
                       Effective pixel size used in propagator generation (m).
            depth: float
                     True propagation distance (m).
                 magnified_depth: float
                     Effective propagation distance used in propagator generation (m).
            propagation_method: str
                                Propagation model used to generate propagator
                                ('angular_spectrum' or 'fresnel').
            correct_pixel_size: bool
                                True if effective-magnification pixel-size correction
                                was applied when generating this propagator.
            source_distance: float or None
                             Source-to-camera distance used for pixel-size correction.
        """
        self.propagator = propagator
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.magnified_pixel_size = magnified_pixel_size
        self.depth = depth
        self.magnified_depth = magnified_depth
        self.propagation_method = propagation_method
        self.correct_pixel_size = correct_pixel_size
        self.source_distance = source_distance
        self.shape = np.shape(propagator)

    def has_attributes(
        self,
        wavelength=None,
        pixel_size=None,
        magnified_pixel_size=None,
        depth=None,
        magnified_depth=None,
        propagation_method=None,
        correct_pixel_size=None,
        source_distance=None,
    ):
        """Checks if the propagator has the specified attributes.

        Keyword arguments:
            wavelength      : float
                              The wavelength to check against (m).
            pixel_size      : float
                              The pixel size to check against (m).
            depth           : float
                              The depth to check against (m).
        Returns:
            bool            : True if the propagator has all the specified attributes,
                              False otherwise.
        """

        if wavelength is not None and self.wavelength != wavelength:
            return False
        if pixel_size is not None and self.pixel_size != pixel_size:
            return False
        if (
            magnified_pixel_size is not None
            and self.magnified_pixel_size != magnified_pixel_size
        ):
            return False
        if depth is not None and self.depth != depth:
            return False
        if magnified_depth is not None and self.magnified_depth != magnified_depth:
            return False
        if (
            propagation_method is not None
            and self.propagation_method != propagation_method
        ):
            return False
        if (
            correct_pixel_size is not None
            and self.correct_pixel_size != correct_pixel_size
        ):
            return False
        if source_distance is not None and self.source_distance != source_distance:
            return False

        return True
