import numpy as np


class Propagator:
    """Represents a propagator for wavefront propagation in holography.

    Attributes:
        propagator: 2D complex numpy array containing the propagator.
        wavelength: The wavelength of the light used in the propagation (m).
        pixel_size: The size of the pixels in the hologram (m).
        depth: Propagation distance (m).
    """

    wavelength = None
    pixel_size = None
    depth = None
    propagator = None
    shape = None
    geometry = None

    def __init__(
        self, propagator, wavelength=None, pixel_size=None, depth=None, geometry=None
    ):
        """
        Initialises the propagator with a specific wavelength and pixel size.
        The depth is optional and can be used for some propagators.
        """
        self.propagator = propagator
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.depth = depth
        self.geometry = geometry
        self.shape = np.shape(propagator)

    def has_attributes(
        self, wavelength=None, pixel_size=None, depth=None, geometry=None
    ):
        """Checks if the propagator has the specified attributes.

        Keyword arguments:
            wavelength      : float
                              The wavelength to check against (m).
            pixel_size      : float
                              The pixel size to check against (m).
            depth           : float
                              The depth to check against (m).
            geometry        : str
                              The geometry to check against ('plane' or 'point').
        Returns:
            bool            : True if the propagator has all the specified attributes,
                              False otherwise.
        """

        if wavelength is not None and self.wavelength != wavelength:
            return False
        if pixel_size is not None and self.pixel_size != pixel_size:
            return False
        if depth is not None and self.depth != depth:
            return False
        if geometry is not None and self.geometry != geometry:
            return False

        return True
