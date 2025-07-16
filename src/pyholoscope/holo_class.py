## -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy in Python

The Holo Class provides an object-oriented interface to most of the
PyHoloscope functionality.
"""

import numpy as np
import time
import warnings

from pyholoscope.utils import (
    circ_cosine_window,
    circ_window,
    square_cosine_window,
    dimensions,
)
from pyholoscope.off_axis import (
    off_axis_find_mod,
    off_axis_find_crop_radius,
    off_axis_demod,
)
from pyholoscope.focusing import propagator, refocus, find_focus, refocus_stack
from pyholoscope.general import pre_process
from pyholoscope.phase_proc import relative_phase
from pyholoscope.prop_lut import PropLUT
from pyholoscope.roi import Roi


# Check if cupy is available
try:
    import cupy as cp

    cuda_available = True
except:
    cuda_available = False

# Check if numba is available
try:
    import numba
    from pyholoscope.focusing_numba import propagator_numba
    testProg = propagator_numba(
        (int(6), int(6)),
        float(1.0),
        float(1.0),
        float(1.0),
        geometry="plane",
        precision="single",
    )  # Run the JIT once for speed
    numba_available = True

except:
    numba_available = False


class Holo:
    # Processing pipeline
    INLINE = 1
    OFF_AXIS = 2
    modes = ["NONE", "INLINE", "OFF AXIS"]
    INLINE_MODE = 1  # deprecated, kept for backwards compatibility
    OFFAXIS_MODE = 2  # deprecated, kept for backwards compatibility

    # Off-axis crop mask types
    CUSTOM = 1
    CIRCLE = 2
    CIRCLE_COSINE = 3

    # For off-axis holography, these are generated if needed from the
    # background and normalisation holograms later
    background_field = None
    background_abs = None
    background_angle = None
    normalise_field = None
    normalise_abs = None
    normalise_angle = None
    crop_window = None

    # Standard image type
    image_type = "float32"
    
    propagator = None

    def __init__(self, mode=None, wavelength=None, pixel_size=None, **kwargs):
        self.mode = mode
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.oa_pixel_size = pixel_size

        self.use_numba = kwargs.get("numba", True)
        self.cuda = kwargs.get("cuda", True)

        # Numerical refocusing
        self.depth = kwargs.get("depth", 0)
        self.set_background(kwargs.get("background", None))
        self.set_normalise(kwargs.get("normalise", None))
        self.geometry = kwargs.get("geometry", 'point')

        # Phase
        self.relative_amplitude = kwargs.get("relative_amplitude", False)

        # Widowing
        self.auto_window = kwargs.get("auto_window", False)
        self.post_window = kwargs.get("post_window", False)
        self.window = kwargs.get("window", None)
        self.window_shape = kwargs.get("window_shape", "square")
        self.window_radius = kwargs.get("window_radius", None)
        self.window_thickness = kwargs.get("window_thickness", 10)

        # Autofocus
        self.find_focus_method = kwargs.get("find_focus_method", "Brenner")
        self.find_focus_roi = kwargs.get("find_focus_roi", None)
        self.find_focus_margin = kwargs.get("find_focus_margin", None)
        self.find_focus_coarse_search_interval = kwargs.get(
            "find_focus_coarse_search_interval", None
        )
        self.find_focus_depth_range = kwargs.get("find_focus_depth_range", (0, 1))

        # Off-axis demodulation
        self.crop_centre = kwargs.get("crop_centre", None)
        self.crop_radius = kwargs.get("crop_radius", None)
        self.return_fft = kwargs.get("return_fft", False)
        self.crop_mask = kwargs.get("crop_mask", None)
        self.custom_crop_window = kwargs.get("custom_crop_window", None)
        self.crop_window_skin_thickness = kwargs.get("crop_window_skin_thickness", 10)

        # Phase
        self.relative_phase = kwargs.get("relative_phase", False)
        self.stable_phase = kwargs.get("stable_phase", False)
        self.stable_roi = kwargs.get("stable_roi", False)

        # Display
        self.invert = kwargs.get("invert", False)
        self.refocus = kwargs.get("refocus", False)
        self.downsample = kwargs.get("downsample", 1)

        # GPU
        self.cuda_available = cuda_available

        # Image data type
        self.set_precision(kwargs.get("precision", "single"))

    def __process_inline(self, img):
        """Process an inline hologram image, img, using the currently selected
        parameters.
        """

        # If we doing auto_window, and we either don't have
        # a window, or it is the wrong size, we make a new window
        self.update_auto_window(img)

        imgPreprocessed = pre_process(
            img,
            downsample=self.downsample,
            window=self.window,
            background=self.background,
            normalise=self.normalise,
            precision=self.precision,
        )

        # If the propagator is not the correct one for the current parameters, regenerate it
        if self.propagator is None or not self.propagator.has_attributes(wavelength=self.wavelength,
                                          pixel_size=self.pixel_size,
                                          depth=self.depth) is False:
            self.update_propagator(img)

        # Numerical refocusing
        imgOut = refocus(
            imgPreprocessed, self.propagator, cuda=(self.cuda and cuda_available)
        )
        if imgOut is None:
            warnings.warn("Output from refocusing was None.")
            return None

        # Post refocusing processing
        if self.post_window is True and self.window is not None:
            imgOut = pre_process(imgOut, window=self.window, precision=self.precision)

        if self.invert is True:
            imgOut = np.max(imgOut) - imgOut

        return imgOut

    def __process_off_axis(self, img):
        """Process an off-axis hologram image using the currently selected parameters."""

        assert self.crop_centre is not None, "Off-Axis demodulation frequency not defined."
        assert self.crop_radius is not None, "Off-Axis demodulation radius not defined."

        # Removes the off-axis modulation to obtain complex image
        demod = off_axis_demod(
            img,
            self.crop_centre,
            self.crop_radius,
            mask=self.cropWindow,
            return_fft=self.return_fft,
            cuda=self.cuda,
        )

        if demod is None:
            warnings.warn("Output from off-axis demodulation was None.")
            return None

        # If return_fft is True, off_axis_demod returns the demodulated image and the FFT as a tuple. If we
        # have been asked for the FFT we pull this out and return it, otherwise 'demod' is the demodulated image
        # and we continue
        if self.return_fft:
            return demod[1]

        # Relative phase means to subtract the phase from the background image
        if self.relative_phase == True:
            if self.background_field is not None:
                demod = relative_phase(demod, self.background_field)
            elif self.background is not None:
                self.off_axis_background_field()

            else:
                warnings.warn(
                    "Relative phase requested but no background field available, call off_axis_background_field() to create this first."
                )

        if demod is None:
            warnings.warn("Output from off-axis relative phase was None.")
            return None

        # If we have are doing auto_window, and we either don't have
        # a window, or it is the wrong size, we make a new window
        self.update_auto_window(img)

        # Off axis demodulation changes the pixel size,
        # so here we calculate the corrected pizel size
        if self.pixel_size is not None:
            self.oa_pixel_size = (
                self.pixel_size / float(np.shape(demod)[0]) * float(np.shape(img)[0])
            )

        # Apply background, normalisation, windowing, downsampling
        if self.relative_amplitude:
            background = self.background_abs
        else:
            background = None
        
        demod = pre_process(
            demod,
            downsample=self.downsample,
            window=self.window,
            background=background,
            normalise=self.normalise_abs,
            precision=self.precision,
        )

        # Numerical refocusing
        if self.refocus is True:
            # Check the propagator is valid, otherwise recreate it
            if self.propagator is None or not self.propagator.has_attributes(
                depth=self.depth,
                wavelength=self.wavelength,
                pixel_size=self.oa_pixel_size,
            ):
                self.update_propagator(demod)

            # Refocus
            demod = refocus(demod, self.propagator, cuda=(self.cuda and cuda_available))

            if demod is None:
                warnings.warn("Output from off-axis refocusing was None.")
                return None

        # Post refocusing processing
        if demod is not None:
            if (
                self.post_window is True
                and self.auto_window is True
                and self.window is not None
            ):
                demod = pre_process(demod, window=self.window, precision=self.precision)

            if self.invert is True:
                demod = np.max(demod) - demod

        if demod is None:
            warnings.warn("Output from off-axis processing was None.")
            return None

        return demod

    def __apply_window(self, img):
        """Applies the current window to a hologram 'img'."""
        img = pre_process(img, window=self.window)
        return img

    def __str__(self):
        return (
            "PyHoloscope Holo Class. Wavelength: "
            + str(self.wavelength)
            + ", Pixel Size: "
            + str(self.pixel_size)
        )

    ###############################################################################
    ####################################### API ###################################

    def process(self, img):
        """Process a hologram using the currently selected parameters.
        Calls _process_inline or __process_off_axis depending  on mode.
        """

        # If we are refocusing we must have a wavelength, pixel size and depth specified
        if self.mode == self.INLINE or (
            self.mode == self.OFF_AXIS and self.refocus == True
        ):
            assert self.pixel_size is not None, "Pixel size not specified."
            assert self.wavelength is not None, "Wavelength not specified."
            assert self.depth is not None, "Refocus depth not specified."

        if img is None:
            warnings.warn("Image provided to process was None, output will be None.")
            return None

        assert img.ndim == 2, "Input must be a 2D numpy array."

        if self.mode == self.INLINE_MODE or self.mode == self.INLINE:
            return self.__process_inline(img)
        elif self.mode == self.OFFAXIS_MODE or self.mode == self.OFF_AXIS:
            return self.__process_off_axis(img)
        else:
            raise Exception("Invalid processing mode.")

    def set_mode(self, mode):
        self.mode = mode

    def set_refocus(self, refocus):
        self.refocus = refocus

    ############### PHYSICAL PARAMETERS ##########################################

    def set_wavelength(self, wavelength):
        """Set the wavelength of the hologram"""
        self.wavelength = wavelength

    def set_pixel_size(self, pixel_size):
        """Set the size of pixels in the raw hologram"""
        self.pixel_size = pixel_size

    ########### BACKGROUND AND FLAT-FIELDING #####################################

    def set_background(self, background):
        """Set the background hologram. Use None to remove background."""
        self.clear_background()
        if background is not None:
            self.background = background.astype(self.image_type)
        else:
            self.background = None

    def set_normalise(self, normalise):
        """Set the normalisation hologram. Use None to remove normalisation."""
        self.clear_normalise()
        if normalise is not None:
            self.normalise = normalise.astype(self.image_type)

    def clear_background(self):
        """Remove existing background hologram."""
        self.background = None
        self.background_field = None
        self.background_abs = None
        self.background_angle = None

    def clear_normalise(self):
        """Remove existing normalisation hologram."""
        self.normalise = None
        self.normalise_field = None
        self.normalise_abs = None
        self.normaliseAngle = None

    def set_relative_amplitude(self, boolean):
        """Sets whether or not to calculate relative phase in off-axis holography."""
        assert boolean == True or boolean == False, (
            "Argument of set_relative_amplitude must be True or False"
        )
        self.relative_amplitude = boolean

    ############################# WINDOW #####################################

    def create_window(self, imgSize, radius, skinThickness, shape="square"):
        """Creates and stores the window used for pre or post processing.

        Arguments:
            imgSize       :  the size of the window array, must be the same as the hologram it will be
                             applied to. Either provide a 2D numpy array, in which case the window will
                             be created to match the size of this, provide an int, in which case the window
                             will be a square of this size or a tuple of (width, height).
            radius        :  the size of the  transparent part of the window, for 'circle' this is the
                             radius, for 'square' this is half the side length. For 'circle' provide
                             an int, for 'square' either provide an int (resulting in a square window)
                             or a tuple of (width, height) for rectangular window.
            skinThickness :  The number of pixels inside the window over which it transitions from
                             opaque to transparent.

        Keyword Arguments:
            shape         :  [Optional] window shape, 'circle' or 'square' (defualt).
        """

        if shape == "circle":
            self.window = circ_cosine_window(
                imgSize, radius, skinThickness, dataType=self.image_type
            )
        elif shape == "square":
            self.window = square_cosine_window(
                imgSize, radius, skinThickness, dataType=self.image_type
            )

    def set_window(self, window):
        """Sets the window to a pre-generated 'window', a 2D numpy array."""
        self.clear_window()
        if window is not None:
            self.window = window.astype(self.image_type)

    def clear_window(self):
        """Removes existing window, equivalent to set_window(None)"""
        self.window = None

    def set_auto_window(self, auto_window):
        """Sets whether or not use auto create a window (boolean)."""
        assert auto_window == True or auto_window == False, (
            "set_auto_window must be True or False"
        )
        self.auto_window = auto_window

    def set_post_window(self, post_window):
        """Sets whether or not to apply the window after refocusing (boolean)."""
        assert post_window == True or post_window == False, (
            "set_post_window must be True or False"
        )
        self.post_window = post_window

    def update_auto_window(self, img):
        """Create or re-create the automatic window using current parameters.
        Provide an 'img', a 2D numpy array of the same size as the image to
        be processed.
        """

        imHeight = np.shape(img)[0]
        imWidth = np.shape(img)[1]

        if self.auto_window == True:
            if self.window is None:
                regenWindow = True
            elif (
                np.shape(self.window)[0] != np.shape(img)[0] / self.downsample
                or np.shape(self.window)[1] != np.shape(img)[1] / self.downsample
            ):
                regenWindow = True
            else:
                regenWindow = False

            if regenWindow:
                if self.window_radius is None:
                    window_radiusX = int(imWidth / 2)
                    window_radiusY = int(imHeight / 2)
                else:
                    window_radiusX, window_radiusY = dimensions(self.window_radius)

                self.create_window(
                    (int(imWidth / self.downsample), int(imHeight / self.downsample)),
                    (
                        int(window_radiusX / self.downsample),
                        int(window_radiusY / self.downsample),
                    ),
                    self.window_thickness / self.downsample,
                    shape=self.window_shape,
                )

    def set_window_shape(self, window_shape):
        """Sets the auto window shape, 'cicle' or 'square'."""
        if window_shape == "circle" or window_shape == "square":
            self.window_shape = window_shape
        else:
            raise Exception("Invalid window shape.")

    def set_window_radius(self, window_radius):
        """Sets the auto window radius."""
        self.window_radius = window_radius

    def set_window_thickness(self, window_thickness):
        """Sets the auto window edge thickness."""
        self.window_thickness = window_thickness

    ################## OFF AXIS DEMODULTION ######################################

    def set_off_axis_mod(self, crop_centre, crop_radius):
        """Sets the location of the frequency domain position of the OA modulation.

        Arguments:
            crop_centre  : tuple of (x,y)
            crop_radius  : radius
        """
        self.crop_centre = crop_centre
        self.crop_radius = crop_radius

    def set_crop_centre(self, centre):
        """Set the location of the modulation frequency in frequency domain.
        'centre' is a tuple of (x,y).
        """
        self.crop_centre = dimensions(centre)

    def set_crop_radius(self, radius):
        """Set the size of the region to extract in frequency domain to demodulate."""
        self.crop_radius = dimensions(radius)

    def set_return_FFT(self, return_fft):
        """Sets whether the FFT, rather than the demodualted image, is returned in OAH.
        Set True to obtain FFT, False to obtain image.
        """
        self.return_fft = return_fft

    def auto_find_off_axis_mod(self, maskFraction=0.1):
        """Detect the modulation location in frequency domain. maskFraction
        is the size of a mask applied to the centre of the FFT to prevent
        the d.c. from being detected.
        """
        if self.background is not None:
            self.crop_centre = off_axis_find_mod(self.background, maskFraction=0.1)
            self.crop_radius = off_axis_find_crop_radius(self.background)

    def calib_off_axis(self, hologram=None, mask_fraction=0.1):
        """Detect the modulation location in frequency domain using the
        background or a provided hologram.
        """

        if hologram is None:
            hologram = self.background

        if hologram is None:
            raise Exception(
                "Calib_off_axis requires a calibration image, either provided as an argument or from a previously set background."
            )

        self.crop_centre = off_axis_find_mod(hologram, mask_fraction=mask_fraction)
        self.crop_radius = off_axis_find_crop_radius(hologram)

        if self.background is not None:
            self.off_axis_background_field()
        if self.normalise is not None:
            self.off_axis_normalise_field()

        if self.crop_mask == self.CIRCLE:
            self.cropWindow = circ_window(
                (self.crop_radius[0] * 2, self.crop_radius[1] * 2), self.crop_radius
            )
        elif self.crop_mask == self.CIRCLE_COSINE:
            self.cropWindow = circ_cosine_window(
                (self.crop_radius[0] * 2, self.crop_radius[1] * 2),
                self.crop_radius,
                self.crop_window_skin_thickness,
            )
        elif self.crop_mask == self.CUSTOM:
            self.cropWindow = self.custom_crop_window
        else:
            self.cropWindow = None

    def off_axis_background_field(self):
        """Demodulate the background hologram."""
        assert self.background is not None, "Background hologram not provided."
        assert self.crop_centre is not None, "Demodulation centre not provided"
        assert self.crop_radius is not None, "Demodulation radius not provided."
        self.background_field = off_axis_demod(
            self.background, self.crop_centre, self.crop_radius
        )
        self.background_abs = np.abs(self.background_field)  # Store these now for speed
        self.backgroundPhase = np.angle(self.background_field)

    def off_axis_normalise_field(self):
        """Demodulate the background hologram."""
        assert self.background is not None, "Background hologram not provided."
        assert self.crop_centre is not None, "Demodulation centre not provided"
        assert self.crop_radius is not None, "Demodulation radius not provided."
        self.normalise_field = off_axis_demod(
            self.normalise, self.crop_centre, self.crop_radius
        )
        self.normalise_abs = np.abs(self.normalise_field)  # Store these now for speed
        self.normalisePhase = np.angle(self.normalise_field)

    ##################### PHASE #############################################

    def set_stable_ROI(self, roi):
        """Set the location of the the ROI used for maintaining a constant
        background phase, i.e. this should be a background region of the image.
        The roi should be an instance of the Roi class.
        """
        assert isinstance(roi, Roi), ("Argument must be an instance of Roi.")
        self.stable_roi = roi

    def set_relative_phase(self, relative_phase):
        """Sets whether or not to use relative phase, i.e. phase
        is relative to the phase of the background hologram.
        """

        assert relative_phase == True or relative_phase == False, (
            "Argument of set_relative_phase must be True or False"
        )
        self.relative_phase = relative_phase

    ##################### REFOCUSING #########################################

    def set_depth(self, depth):
        """Set the depth for numerical refocusing"""
        self.depth = depth

    def update_propagator(self, img):
        """Create or re-create the propagator using current parameters. img
        should be an 2D numpy array of the same size as the images to be processed.
        """

        self.propagatorWavelength = self.wavelength
        self.propagatorDepth = self.depth

        if self.mode == self.INLINE_MODE:
            self.propagatorpixel_size = self.pixel_size * self.downsample
            downsample = self.downsample
        else:
            self.propagatorpixel_size = self.oa_pixel_size
            downsample = 1

        propWidth = int(np.shape(img)[1] / downsample / 2) * 2
        propHeight = int(np.shape(img)[0] / downsample / 2) * 2

        self.propagator = propagator(
            (propHeight, propWidth),
            self.wavelength,
            self.pixel_size,
            self.depth,
            precision=self.precision,
            geometry = self.geometry,
            use_numba=self.use_numba,
        )

        # If using CUDA we send propagator to GPU now to speed up refocusing later
        if self.cuda and cuda_available:
            self.propagator.propagator = cp.array(self.propagator.propagator)

    def set_downsample(self, downsample):
        """Set the downsample factor. This will cause the propagator to be
        recreated when next needed, call update_propagator to force this immediately.
        """
        if downsample != self.downsample:
            self.propagator = None  # Force to be recreated when needed

        self.downsample = downsample

    ############### AUTO FOCUS ###################################################

    def set_find_focus_parameters(self, **kwargs):
        """Sets the parameters used by the find_focus method.

        Keyword Arguments:
            depthRange   : double
                           tuple of (min, max) depths to search within in m.
            roi          : instance of Roi
                           area to assess focus within, default is None in which
                           case all of image is used.
            method       : str
                           focus metric to use.
            margin       : int
                           if specified only the Roi and a margin will be
                           refocused. If None (default) the whole image will be
                           refocused regardless. Has no effect if roi not specified.
            coarseSearchInterval  : Number of points to check explicitly before
                                    optimising. Default is None, in which case
                                    this is not performed.


        """
        self.find_focus_depth_range = kwargs.get("depthRange", (0, 0.1))
        self.find_focus_roi = kwargs.get("roi", None)
        self.find_focus_method = kwargs.get("method", "Brenner")
        self.find_focus_,margin = kwargs.get("margin", None)
        self.coarseSearchInterval = kwargs.get("coarseSearchInterval", None)

    def find_focus(self, img):
        """Automatically finds the best focus position for hologram 'img'
        using parameters previously defined (such as wavelength) as well
        as by set_find_focus_parameters().

        Arguments:
            img         : ndarray
                          2D array containing hologram

        Returns:
            float       : optimal refocus depth
        """

        args = {
            "background": self.background,
            "window": self.window,
            "roi": self.find_focus_roi,
            "margin": self.find_focus_margin,
            "numba": numba_available and self.use_numba,
            "cuda": cuda_available and self.cuda,
            "propagatorLUT": self.propagatorLUT,
            "coarseSearchInterval": self.find_focus_coarse_search_interval,
        }

        return find_focus(
            img,
            self.wavelength,
            self.pixel_size,
            self.find_focus_depth_range,
            self.find_focus_method,
            precision=self.precision,
            **args,
        )

    def auto_focus(self, img, **kwargs):
        """Finds the best focus, allowing all relevant paramters to be specified as keyword arguments.

        Arguments:
            img         : ndarray
                          2D array containing hologram
        Keyword Arguments:
            depthRange  : tuple
                          depths to search within in m, default is (0,1).
            method      : str
                          focus metric to use, default is 'Brenner'.
            background  : ndarray
                          background hologram, default is None.
            window      : ndarray
                          window to apply to image, default is None.
            numba       : bool
                          whether to use numba JIT, default is True.
            cuda        : bool
                          whether to use GPU if available, default is True.        `
            roi         : instance of Roi
                          area to assess focus within, default is None in which
                          case all of image is used.
            margin      : int or None
                          if specified only the Roi and a margin will be
                          refocused. If None (default) the whole image will be
                          refocused regardless. Has no effect if roi not specified.
            propLUT     : instance of PropLUT or None
                          propagator LUT to use, default is None in which case
                          no LUT is used.
            coarseSearchInterval : int or None
                                   Number of points to check explicitly before
                                   optimising. Default is None, in which case
                                   this is not performed.
            precision   : str
                          precision to use, 'single' or 'double', default is 'single'.





        """
        focusDepth = find_focus(
            img,
            self.wavelength,
            self.pixel_size,
            kwargs.get("depthRange", (0, 1)),
            kwargs.get("method", "Brenner"),
            background=self.background,
            window=self.window,
            numba=self.use_numba and numba_available,
            cuda=self.cuda and cuda_available,
            roi=kwargs.get("roi", None),
            margin=kwargs.get("margin", None),
            propLUT=kwargs.get("propagatorLUT", None),
            coarseSearchInterval=kwargs.get("coarseSearchInterval", None),
            precision=self.precision,
        )
        return focusDepth

    def make_propagator_LUT(self, img, depthRange, nDepths):
        """Creates a LUT of propagators for faster finding of focus."""
        self.propagatorLUT = PropLUT(
            np.shape(img)[0],
            self.wavelength,
            self.pixel_size,
            depthRange,
            nDepths,
            numba=(numba_available and self.use_numba),
            precision=self.precision,
        )

    def clear_propagator_LUT(self):
        """Deletes the LUT of propagators."""
        self.propagatorLUT = None

    ################### DEPTH STACK ##############################################

    def depth_stack(self, img, depthRange, nDepths):
        """Create a depth stack using current parameters, producing a set of
        'nDepths' refocused images equally spaced within depthRange.
        Arguments:
            img        : ndarray
                         hologram
            depthRange : tuple
                         depths to focus to: (min depth, max depth)
            nDepths    : int
                         number of depths to create images for within depthRange

        Returns:
            ndarray    : 3D array containing images refocused to different depths.
        """

        if self.mode == self.INLINE_MODE:
            preBackground = self.background
            postBackground = None
        else:
            preBackground = None
            postBackground = None
        args = {
            "background": self.background,
            "window": self.window,
            "numba": numba_available and self.use_numba,
        }

        return refocus_stack(
            img,
            self.wavelength,
            self.pixel_size,
            depthRange,
            nDepths,
            precision=self.precision,
            **args,
        )


    ########################### GENERAL SETTINGS ################################

    def set_use_cuda(self, useCuda):
        """Set whether to use GPU if available, useCuda is True to use GPU or
        False to not use GPU.
        """
        self.cuda = useCuda

    def set_use_numba(self, use_numba):
        """Set whether to use Numba JIT if available, use_numba is True to use
        Numba or False to not use Numba.
        """
        self.use_numba = use_numba

    def set_precision(self, precision):
        """Sets whether to use single or double precision."""
        assert precision == "single" or precision == "double", (
            "Precision must be 'single' or 'double'."
        )
        self.precision = precision
        if self.precision == "double":
            self.imType = "float64"
        else:
            self.imType = "float32"
