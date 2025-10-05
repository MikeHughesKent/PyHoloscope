# -*- coding: utf-8 -*-
"""
PyHoloscope - Fast Holographic Microscopy for Python

This file contains functions related to numerical refocusing.

"""

import math
import numpy as np

try:
    import cupy as cp

    cuda_available = True
except:
    cuda_available = False

try:
    from pyholoscope.focusing_numba import propagator_numba

    numba_available = True
except:
    numba_available = False


from PIL import Image
import cv2 as cv
import scipy

from pyholoscope.focus_stack import FocusStack
from pyholoscope.focusing_numba import propagator_numba
from pyholoscope.roi import Roi
import pyholoscope.general
from pyholoscope.utils import dimensions
from pyholoscope.propagator import Propagator


def propagator(
    grid_size,
    wavelength,
    pixel_size,
    depth,
    geometry="plane",
    precision="single",
    use_numba=True,
):
    """Creates Fourier domain propagator for angular spectrum method.
    Returns the propagator as an instance of Propagator. Generation is sped up
    by only calculating top left quadrant and then duplicating
    (with flips) to create the other quadrants.

    Wavelength, pixel_size and depth are all specified in the same units.

    Arguments:
        grid_size   : float or (float, float)
                     size of square image (in pixels) to refocus, or
                     if tuple of (float, float), size of rectangular image as
                     height x width
        wavelength : float
                     wavelength of light
        pixel_size  : float
                     physical size of pixels
        depth      : float
                     refocus distance

    Keyword Arguments:
        geometry   : str
                     'plane' (default) or 'point'
        precision  : str
                     numerical precision of output, 'single' (default)
                     or 'double'
        use_numba   : boolean
                     if True, uses Numba version of functions (default is False)

    Returns:
        numpy.ndarray : 2D complex array, propagator

    """

    if precision == "double":
        data_type = "complex128"
    elif precision == "single":
        data_type = "complex64"
    else:
        raise Exception(f"Invalid precision {precision}, must be 'single' or 'double'.")

    grid_height, grid_width = dimensions(grid_size)
    if numba_available and use_numba:
        prop = propagator_numba(
            (grid_height, grid_width),
            wavelength,
            pixel_size,
            depth,
            geometry,
            precision,
        )
        return Propagator(
            propagator=prop,
            wavelength=wavelength,
            pixel_size=pixel_size,
            depth=depth,
            geometry=geometry,
        )

    grid_height, grid_width = dimensions(grid_size)

    centre_x = int(grid_width // 2)
    centre_y = int(grid_height // 2)

    # Physical size of hologram in real units
    width = grid_width * pixel_size
    height = grid_height * pixel_size

    # Grid points to generate one quadrant of propagator on
    (xM, yM) = np.meshgrid(range(centre_x + 1), range(centre_y + 1))

    # Bins size of FFT (i.e. pixel size of FFT in inverse distance)
    delta0x = 1 / width
    delta0y = 1 / height
   
    # Generate one quadrant of the propagator
    if geometry == "point":
        u = delta0x * xM
        v = delta0y * yM
        prop_corner = np.exp(1j * math.pi * wavelength * depth * (u**2 + v**2))

    elif geometry == "plane":
        alpha = wavelength * xM / width
        beta = wavelength * yM / height
        prop_corner = np.exp(
            (-1j * 2 * math.pi * depth * np.sqrt(1 - alpha**2 - beta**2) / wavelength)
        )
        prop_corner[alpha**2 + beta**2 > 1] = 0

    else:
        raise Exception("Invalid geometry.")

    # Array to hold full propagator
    prop = np.zeros((grid_height, grid_width), dtype=data_type)

    # Duplicate the top left quadrant into the other three quadrants as
    # this is quicker then explicitly calculating the values
    prop[: centre_y + 1, : centre_x + 1] = prop_corner  # top left
    prop[: centre_y + 1, centre_x + grid_width % 2 :] = np.flip(
        prop_corner[:, 1:], 1
    )  # top right
    prop[centre_y + grid_height % 2 :, :] = np.flip(
        prop[1 : centre_y + 1, :], 0
    )  # bottom half

    prop_obj = Propagator(
        propagator=prop,
        wavelength=wavelength,
        pixel_size=pixel_size,
        depth=depth,
        geometry=geometry,
    )
    return prop_obj





def refocus(img, propagator, **kwargs):
    """Refocuses a hologram using the angular spectrum method.
    Takes a hologram 'hologram' wich may be a real or
    complex 2D numpy array (with any pre-processing
    such as background removal already performed) and a pre-computed 'propagator'
    which can be generated using the function 'propagator()'.

    Arguments:
        img           : ndarray
                        2D numpy array, raw hologram.
        propagator    : instance of Propagator
                        propagator object, containing the propagator as a
                        2D numpy array, as returned from propagator().

    Keyword Arguments:
        background    : numpy.ndarray or None
                        background image to subtract prior to refocus, same shape as img (default is None)
        normalise     : numpy.ndarray or None
                        normalisation image to divide by prior to refocus, same shape as img (default is None)
        window        : numpy.ndarray or None
                        spatial window to apply to image prior to refocus, same shape as img (default is None)
        downsample    : int or None
                        downsample factor, if not None, the image will be downsampled by this factor prior to refocus (default is None)
        fourier_domain : boolean
                        if True then img is assumed to be already the
                        FFT of the hologram, useful for speed when performing
                        multiple refocusing of the same hologram. (default is
                        False)
        cuda          : boolean
                        if True (default) GPU will be used if available.
        others        : pass any keyword arguments from pre_process() to
                        apply this pre-processing prior to refocusing

    Returns:
        numpy.ndarray : 2D complex array, refocused image
    """

    img_is_fourier = kwargs.pop("fourier_domain", False)
    cuda = kwargs.pop("cuda", True)

    # If we were sent the propagator on the CPU, push to GPU now
    if (
        cuda is True
        and cuda_available is True
        and type(propagator.propagator) is np.ndarray
    ):
        prop = cp.array(propagator.propagator)
    else:
        prop = propagator.propagator

    # If we have been sent the FFT of image, used when repeatedly calling refocus
    # (for example when finding best focus) we don't need to do FFT or shift for speed
    if img_is_fourier:
        if cuda is True and cuda_available is True:
            if type(img) is np.ndarray:
                img = cp.array(img)
            return cp.asnumpy(cp.fft.ifft2(img * prop))
        else:
            return scipy.fft.ifft2(img * prop)

    else:  # If we are sent the spatial domain image
        c_hologram = pyholoscope.pre_process(img, **kwargs)
        assert np.shape(c_hologram) == propagator.shape, (
            f"Propagator is the wrong size {propagator.shape} compared with hologram {np.shape(c_hologram)}."
        )

        if cuda is True and cuda_available is True:
            if type(c_hologram) is np.ndarray:
                c_hologram = cp.array(c_hologram)
            return cp.asnumpy(cp.fft.ifft2(cp.fft.fft2(c_hologram) * prop))
        else:
            return scipy.fft.ifft2(scipy.fft.fft2(c_hologram) * prop)


def correct_curvature(img, wavelength, pixel_size, source_distance):
    """Corrects for spherical wavefront curvature in hologram, assuming a point source.

    Arguments:
        img             : ndarray
                          2D numpy array, raw hologram.
        wavelength     : float
                          wavelength of light source
        pixel_size      : float
                          real pixel size of hologram
        source_distance: float
                          distance from point source to camera plane

    Returns:
        numpy.ndarray : 2D complex array, corrected hologram

    """

    (h, w) = np.shape(img)
    (xM, yM) = np.meshgrid(range(w), range(h))

    centre_x = int(w // 2)
    centre_y = int(h // 2)

    # Real space coordinates of each pixel relative to centre of image
    x = pixel_size * (xM - centre_x)
    y = pixel_size * (yM - centre_y)

    # Distance from point source to each pixel
    #r = np.sqrt(x**2 + y**2 + source_distance ** 2)
    phase = (x**2 + y**2) / (2 * source_distance)

    correction = np.exp(-1j * 2 * math.pi * phase / wavelength)
    
    return img * correction


def focus_score(img, method):
    """Returns score of how 'in focus' an amplitude image is.
    Score is returned as a float, the lower the better the focus.

    Arguments:
        img          : numpy.ndarray
                       image to score, 2D real array
        method       : str
                       scoring method, options are: 'Brenner', 'Sobel',
                       'SobelVariance', 'Var', 'DarkFcous' or 'Peak'
    Returns:
        float        : focus score
    """

    focus_score = 0

    if method == "Brenner":
        (h, w) = np.shape(img)
        BrennX = np.zeros((h, w))
        BrennY = np.zeros((h, w))
        BrennX[0:-2, :] = img[2:, :] - img[0:-2,]
        BrennY[:, 0:-2] = img[:, 2:] - img[:, 0:-2]
        scoreMap = np.maximum(BrennY**2, BrennX**2)
        focus_score = -np.mean(scoreMap)

    elif method == "Peak":
        focus_score = -np.max(img)

    elif method == "Sobel":
        filtX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        filtY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focus_score = -np.mean(sobel)

    elif method == "SobelVariance":
        filtX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        filtY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        xSobel = scipy.signal.convolve2d(img, filtX)
        ySobel = scipy.signal.convolve2d(img, filtY)
        sobel = xSobel**2 + ySobel**2
        focus_score = -(np.std(sobel) ** 2)

    elif method == "Var":
        focus_score = -np.std(img)

    # https://doi.org/10.1016/j.optlaseng.2020.106195
    elif method == "DarkFocus":
        kernelX = np.array([[-1, 0, 1]])
        kernelY = kernelX.transpose()
        gradX = cv.filter2D(img, -1, kernelX)
        gradY = cv.filter2D(img, -1, kernelY)
        mean, stDev = cv.meanStdDev(gradX**2 + gradY**2)
        focus_score = -(stDev[0, 0] ** 2)

    else:
        raise Exception("Invalid scoring method.")

    return focus_score


def refocus_and_score(
    depth,
    img_fft,
    pixel_size,
    wavelength,
    method,
    score_roi=None,
    prop_lut=None,
    use_numba=False,
    use_cuda=False,
    precision="single",
):
    """Refocuses an image to specificed depth and returns focus score, used by
    findFocus.

    Parameters:
        depth     : float
                    depth to focus to
        img_fft   : ndarray, complex
                    FFT of hologram
        pixel_size: float
                    real pixel size of hologram
        wavelength: float
                    wavelength of light source
        method    : list of str or str,
                    scoring method (see focus_score for list). If a list is
                    provided, a list of scores will be returned, one for each
                    method, otherwise if a single method is provided as
                    a string, then a single score will be returned.

    Keyword Arguments:
        score_roi  : ROI or None
                    region of image to apply scoring to (default is None, in
                    which case it applies to whole image.)
        prop_lut    : prop_lut or None
                    propgator look-up table (default is None, propagator is
                    calculate for each depth)
        use_numba  : boolean
                    if True, uses Numba version of functions (default is False)
        use_cuda   : boolean
                    if True, uses GPU where available
        precision : str
                    'single' (default) or 'double', precision of calculated propagator

    Returns:
        list of floats or float : either a list of scores or a single score.

    """

    # Whether we are using a look up table of propagators or calculating it each time
    if prop_lut is None:
            prop = propagator(
                img_fft,
                wavelength,
                pixel_size,
                depth,
                precision=precision,
                use_numba = use_numba,
            )

    else:
        prop = prop_lut.propagator(depth)

    # We are working with the FFT of the hologram for speed
    refocImg = np.abs(refocus(img_fft, prop, fourier_domain=True, cuda=use_cuda))

    # If we are running focus metric only on a ROI, extract the ROI
    if score_roi is not None:
        refocImg = score_roi.crop(refocImg)

    # If we have list of methods, we return a list of scores, otherwise
    # just a single score
    if isinstance(method, list):
        score = []
        for m in method:
            score.append(focus_score(refocImg, m))
    else:
        score = focus_score(refocImg, method)

    # print(depth, score)

    return score


def find_focus(img, wavelength, pixel_size, depth_range, method, **kwargs):
    """Determines the refocus depth which maximises the focus metric on image.

    To depth score using only a subset of the image, provide an instance of Roi as score_roi. Note that
    the entire image will still be refocused, i.e. this does not provide a speed improvement.

    To refocus only a small region of the image around the ROI (which is faster), provide a margin in margin,
    a region with this margin around the ROI will then be refocused. A pre-computed propagator LUT, an instance of prop_lut, can be
    provided in prop_lut. Note that if margin is specified, the propagator LUT must be of the correct size,
    i.e. the same size as the area to be refocused.

    To perform an initial coarse search to identify the region likely to have the best focus, provide the number of
    search regions to split the search range into in coarse_search_interval.

    Parameters:
        pixel_size  : float
                      real pixel size of hologram
        wavelength  : float
                      wavelength of light source
        depth_range : tuple of (float, float)
                      min and max depths to search between
        method      : str
                      scoring method (see focus_score for list)

    Keyword Arguments:
        background : ndarray or None
                     background image (default is None)
        window     : ndarray or None
                     spatial window (default is None)
        score_roi  : ROI or None
                     region of image to apply scoring to (default is None, in
                     which case it applies to whole image.)
        margin     : int or None
                     if not none, only a region with this margin around the score_roi
                     will be refocused prior to scoring
        prop_lut   : prop_lut or None
                     propagator look up table (default is None)
        coarse_search_interval : int or None
                               number of intervals to divide depth range into
                               for initial search. Default is None, i.e. no
                               initial search.
        use_numba  : boolean
                     if True, uses Numba version of functions (default is False)
        use_cuda   : boolean
                     if True, uses GPU where available

    """
    background = kwargs.get("background", None)
    normalise = kwargs.get("normalise", None)
    window = kwargs.get("window", None)
    score_roi = kwargs.get("roi", None)
    margin = kwargs.get("margin", None)
    prop_lut = kwargs.get("propagatorLUT", None)
    coarse_search_interval = kwargs.get("coarse_search_interval", None)
    use_numba = kwargs.get("numba", False)
    use_cuda = kwargs.get("cuda", False)

    c_hologram = pyholoscope.pre_process(
        img, background=background, normalise=normalise, window=window
    )

    # If a margin is specified, this means we only refocus the ROI plus a
    # margin around it for speed. Define the ROI here.
    if margin is not None and score_roi is not None:
        refocus_roi = Roi(
            score_roi.x - margin,
            score_roi.y - margin,
            score_roi.width + margin * 2,
            score_roi.height + margin * 2,
        )
        refocus_roi.constrain(0, 0, np.shape(img)[0], np.shape(img)[1])
        score_roi = Roi(margin, margin, score_roi.width, score_roi.height)
    else:
        refocus_roi = None

    # If we are only refocusing around a ROI, crop to the ROI + margn
    if refocus_roi is not None:
        cropped_img = refocus_roi.crop(c_hologram)
        score_roi.constrain(0, 0, np.shape(cropped_img)[0], np.shape(cropped_img)[1])
    else:
        cropped_img = c_hologram  # otherwise use whole image

    # Pre-compute the FFT of the hologram since we need this for every trial depth
    img_fft = scipy.fft.fft2(cropped_img)

    if coarse_search_interval is not None:
        startDepth = coarse_focus_search(
            img_fft,
            depth_range,
            coarse_search_interval,
            pixel_size,
            wavelength,
            method,
            score_roi,
            prop_lut,
        )
        interval_size = (depth_range[1] - depth_range[0]) / coarse_search_interval
        min_bound = max(depth_range[0], startDepth - interval_size)
        max_bound = min(depth_range[1], startDepth + interval_size)
        depth_range = [min_bound, max_bound]
    else:
        startDepth = (max(depth_range) - min(depth_range)) / 2

    # Find the depth using optimiser
    depth = scipy.optimize.minimize_scalar(
        refocus_and_score,
        method="golden",
        bracket=depth_range,
        options={"maxiter": 20},
        args=(
            img_fft,
            pixel_size,
            wavelength,
            method,
            score_roi,
            prop_lut,
            use_numba,
            use_cuda,
        ),
    )

    return depth.x


def coarse_focus_search(
    img_fft,
    depth_range,
    num_intervals,
    pixel_size,
    wavelength,
    method,
    score_roi,
    prop_lut,
):
    """An initial check for approximate location of good focus depths prior to a finer search.
    Used by findFocus, see this function for arguments.
    """
    search_depths = np.linspace(depth_range[0], depth_range[1], num_intervals)
    focus_score = np.zeros_like(search_depths)
    for idx, depth in enumerate(search_depths):
        focus_score[idx] = refocus_and_score(
            depth, img_fft, pixel_size, wavelength, method, score_roi, prop_lut
        )

    best_interval = np.argmin(focus_score)
    best_depth = search_depths[best_interval]

    return best_depth


def focus_score_curve(
    img, wavelength, pixel_size, depth_range, nPoints, method, **kwargs
):
    """Produces a plot of focus score against depth, mainly useful for debugging
    erroneous focusing.

    Arguments:
        img        : ndarray
                     2D numpy array, raw hologram
        wavelength : float
                     wavelength of light source
        pixel_size  : float
                     real pixel size of hologram
        depth_range : tuple of (float, float)
                     min and max depths to search between
        nPoints    : int
                     number of points to search between depth_range
        method     : str
                     scoring method (see focus_score for list)

    Keyword Arguments:
        background : ndarray or None
                     background image (default is None)
        window     : ndarray or None
                     spatial window (default is None)
        score_roi  : ROI or None
                     region of image to apply scoring to (default is None, in
                     which case it applies to whole image.)
        margin     : int or None
                     if not none, only a region with this margin around the score_roi
                     will be refocused prior to scoring
        precision  : str
                     'single' (default) or 'double'
        use_numba  : boolean
                     if True, uses Numba version of functions (default is False)
        use_cuda   : boolean
                     if True, uses GPU where available
    """

    background = kwargs.get("background", None)
    normalise = kwargs.get("normalise", None)
    window = kwargs.get("window", None)
    score_roi = kwargs.get("roi", None)
    margin = kwargs.get("margin", None)
    precision = kwargs.get("precision", None)

    c_hologram = pyholoscope.pre_process(
        img, background=background, normalise=normalise, window=window
    )

    if margin is not None and score_roi is not None:
        refocus_roi = Roi(
            score_roi.x - margin,
            score_roi.y - margin,
            score_roi.width + margin * 2,
            score_roi.height + margin * 2,
        )
        refocus_roi.constrain(0, 0, np.shape(img)[0], np.shape(img)[1])
        cropped_img = refocus_roi.crop(c_hologram)
    else:
        cropped_img = c_hologram

    # Do the forwards FFT just once for speed
    c_hologram_fft = scipy.fft.fft2(cropped_img)

    score = []
    depths = np.linspace(depth_range[0], depth_range[1], nPoints)
    for depth in depths:
        score.append(
            refocus_and_score(
                depth,
                c_hologram_fft,
                pixel_size,
                wavelength,
                method,
                score_roi,
                None,
                precision=precision,
            )
        )

    return score, depths


def refocus_stack(
    img,
    wavelength,
    pixel_size,
    depth_range,
    num_depths,
    **kwargs,
):
    """Generates a stack of images by refocusing a hologram to multiple
    depths.

    Parameters:
        wavelength : float
                     wavelength of light source
        pixel_size  : float
                     real pixel size of hologram
        depth_range : tuple of (float, float)
                     min and max depths of stack
        num_depths  : int
                     number of depths to refocus to within depth_range

    Keyword Arguments:
        background : ndarray or None
                     background hologram (default is None)
        window     : ndarray or None
                     spatial window (default is None)
        precision  : str
                     'single' (default) or 'double'
        use_numba  : boolean
                     if True, uses Numba version of functions (default is True)
        use_cuda   : boolean
                     if True, uses GPU where available

    Returns:
        instance of FocusStack
    """

    window = kwargs.get("window", None)
    use_numba = kwargs.get("use_numba", True)
    background = kwargs.get("background", None)
    precision = kwargs.get("precision", "single")
    geometry = kwargs.pop("geometry", "plane")

    depths = np.linspace(depth_range[0], depth_range[1], num_depths)

    # Apply pre-processing and then take 2D FFT
    c_hologram = pyholoscope.pre_process(img, **kwargs)
    c_hologram_fft = scipy.fft.fft2(c_hologram)

    # Tell refocus that we are providing the FFT
    refocus_kwargs = dict()
    refocus_kwargs["fourier_domain"] = True

    img = pyholoscope.pre_process(img, **kwargs)

    imgStack = FocusStack(c_hologram_fft, depth_range, num_depths)

    for idx, depth in enumerate(depths):
        prop = propagator(
            np.shape(img),
            wavelength,
            pixel_size,
            depth,
            precision=precision,
            use_numba=use_numba,
            geometry=geometry,
        )

        imgStack.add_idx(
            refocus(c_hologram_fft, prop, **refocus_kwargs),
            idx,
        )

    return imgStack
