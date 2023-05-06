----------------------------------
Function Reference
----------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^
Classes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: Holo()

Provides object-oriented access to core functionality of PyHoloscope. See `Holo class <holo.html>`_ for details.

.. py:function:: PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths)

Stores a propagator look up table for faster refocusing across multple depths. See `PropLUT class <propLUT.html>`_ for details.

.. py:function:: Roi(x, y, width, height)

Region of interest, a rectangle with top left co-ordinates (``x``, ``y``) with width ``width`` and height ``height``. ``crop`` method is used to extract the ROI
from an image and ``constrain`` method is used to limit co-ordinates to adjust the ROI to fit within an image. See `Roi class <roi.html>`_ for details.


^^^^^^^^^^^^^^^^^^^^^^^^^
General Utility Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: fourier_plane_display(img)

Returns a log-scale Fourier transform of ``img`` for display purposes as 2D real numpy array.

.. py:function:: mean_phase(img)

Returns the mean phase in a complex field ``img``.

.. py:function:: obtain_tilt(img)

Estimates the global tilt in the 2D unwrapped phase (e.g. caused by tilt in coverglass). ``img``
should be unwrapped phase (i.e. real). Returns a 2D real numpy array.

.. py:function:: pre_process(img, background = None, normalise = None, window = None, downsample = 1)
    
Carries out processing steps prior to refocus - background correction, normalisation,
downsampling and  windowing. Also coverts image to either float64 (if input img is real) or
complex128 (if input img is complex). Finally, image is cropped to a square
as non-square images are not currently supported. ``img`` is the raw hologram, a 2D
numpy array, background is a background hologram (2D numpy array) to be subtracted, 
``normalise`` is a background hologram (2D numpy array) to be divided by. ``window`` 
is a window to multiply by (2D numpy array). The image will be downsampled by the
factor given for ``downsample``.

.. py:function:: phase_gradient(img)

Produces a phase gradient (magnitude) image from a complex field ``img``.

.. py:function:: phase_gradient_amp(img)

Returns the ampitude of the phase gradient from a complex field ``img``.

.. py:function:: phase_unwrap(img)

Unwraps phase in 2D from a 2D phase map, ``img``, a 2D real numpy array with wrapped phase. 
Returns a 2D real numpy array.                  

.. py:function:: relative_phase(img, background)

Removes global phase from complex image ``img`` using reference field ``background``.  Returns a 2D complex numpy array.

.. py:function:: relative_phase_ROI(img, roi)
   
Makes the phase in a complex image ``img`` relative to the mean phase in specified ROI ``roi``, an instance of ``Roi``. Returns a 2D complex numpy array.
    
.. py:function:: stable_phase(img, roi = None)

Subtracts the mean phase from the phase map, removing global phase fluctuations. 
Can accept complex ``img``, a complex field, or a real ``img``, which is unwrapped phase in radians. Optionally specify
a region of interest ``roi`` an instance or ``Roi``, in which case the mean phase is calculated from this region only. Returns 
a 2D numpy array, either complex or real depending on the input.
               
.. py:function:: synthetic_DIC(img [, sheerAngle = 0])

Generates a simple, non-rigorous DIC-style image for display from a complex field ``img``. 
The image should appear similar to a relief map, with dark and light regions
correspnding to positive and negative phase gradients along the
shear angle direction ``sheerAngle`` which is specified in radians (default is horizontal, 0 radians). 
The phase gradient is multiplied by the image intensity. 


^^^^^^^^^^^^^^^^^^^
Off-axis Holography
^^^^^^^^^^^^^^^^^^^

.. py:function:: off_axis_demod(hologram, cropCentre, cropRadius, [optional arguments])

Removes spatial modulation from off axis hologram ``hologram``. ``cropCentre`` is the location of
the modulation frequency in the Fourier Domain as tuple (x,y), ``cropRadius`` is the size of
the spatial frequency range to keep around the modulation frequency (in FFT pixels). Returns a 2D complex numpy array.
    
.. py:function:: off_axis_find_mod(hologram [, maskFraction  = 0.1])

Finds the location of the off-axis holography modulation peak in the Fourier transform of ``hologram``. Finds
the peak in the positive x region. Optional argument maskFraction is the fraction of the image masked to avoid detecting the d.c. (default is 0.1). Returns a tuple of (x,y).

.. py:function:: off_axis_find_crop_radius(hologram [, maskFraction  = 0.1])

Estimates the correct off-axis holography crop radius based on modulation peak position in hologram ``hologram``. Optional argument maskFraction is the fraction of the image masked to avoid detecting the d.c. (default is 0.1). Returns a float.
 
.. py:function:: off_axis_predict_mod(wavelength, pixelSize, tiltAngle)

Predicts the location of the modulation peak (i.e. carrer frequency) in the
Fourier transform of a hologram based on the ``wavelength``, camera ``pixelSize`` and the tilt angle of the reference beam ``tiltAngle``.
Returns the distance of the peak from the (dc) of the Fourier transform in pixels.
   
.. py:function:: off_axis_predict_tilt_angle(hologram, wavelength, pixelSize [, maskFraction  = 0.1])

Predicts the reference beam tilt based on the modulation in the hologram ``hologram``
and specified ``wavelength`` and camera ``pixelSize``. Optional argument maskFraction is the fraction of the image masked to avoid detecting the d.c. (default is 0.1). Returns the angle in radians.
    
^^^^^^^^^^^^^^^^^^^^
Numerical Refocusing
^^^^^^^^^^^^^^^^^^^^

.. py:function:: focus_score(img, method)

Returns score of how 'in focus' an image ``img`` is based on selected method ``method``.  
Method options are: Brenner, Sobel, SobelVariance, Var, DarkFcous or Peak.

.. py:function:: coarse_focus_search(imgFFT, depthRange, nIntervals, pixelSize, wavelength, method, scoreROI, propLUT)
Used by find_focus to perform an initial check for approximate location of good focus depths prior to a finer search. 
``imgFFT`` is the 2D Fourier transform of the pre-processed hologram, ``depthRange`` is a tuple of (min depth, max depth) to search over,
``nIntervals`` is the number of search regions to split the depth interval into. ``pixelSize`` and ``wavelength`` are as defined for ``propagator``.
''method'' is the focus scoring method, as defined in ``focus_score``. ``scoreRoi`` is an optional ROI to apply focus score to and ``propLUT`` is an optional propagator
LUT (set either as ``None`` to not use).

.. py:function:: find_focus(img, wavelength, pixelSize, depthRange, method [, background = None, window = None, scoreRoi = None, margin = None, propLUT = None, coarseSearchInterval = None])

Determines the refocus depth which maximises a focus metric on an image ``img`` using a golden section search.
``wavelength`` and ``pixelSize`` are as defined for ``propagator``.
``depthRange`` is a tuple of (min depth, max depth) to search within, in the same units as ``wavelength`` and ``pixelSize``. 
``method`` is the name of the focus scoring method to use, as defined for ``focus_score``.
Optionally specify a ``background`` image as a 2D numpy array and a ``window`` mask image.
To depth score using only a subset of the image, provide an instance of ``Roi`` in ``scoreROI``. Note that 
the entire image will still be refocused, i.e. this does not provide a speed improvement. To refocus only
a small region of the image around the ROI (which is faster), provide a margin in ``margin``, a region with this margin
around the ROI will then be refocused. A pre-computed propagator LUT, an instance of ``PropLUT`` can be 
provided in ``propLUT``. Note that if ``margin`` is specified, the propagator LUT must be of the correct size, i.e. the same size as the area to be refocused.
To perform an initial coarse search to identify the region likely to have the best focus, provide the number of
search regions to split the search range into in ``coarseSearchInterval``.

.. py:function:: focus_score_curve(img, wavelength, pixelSize, depthRange, nPoints, method [, background = None, window = None, scoreROI = None, margin = None])

Produce a plot of focus score against depth, mainly useful for debugging erroneous focusing
Returns a tuple of (numpy vector of scores, numpy vector of dpeth).

.. py:function:: propagator(gridSize, wavelength, pixelSize, depth)
Creates Fourier domain propagator for refocusing using angular spectrum method. ``GridSize``
is the size of the square image (in pixels) that will be refocused, ``wavelength`` is wavelength of light, 
``pixelSize`` is size of camera pixels (as projected onto imaging plane if there is system magnification), 
``depth`` is desired refocus distance. ``wavelength``, ``pixelSize`` and ``depth`` should be in the same units.
Returns a 2D complex numpy array.

.. py:function:: refocus(img, propagator [, imgIsFourier = False, cuda = True])  

Refocuses image using angular spectrum method. Takes a hologram ``hologram`` wich may be a real or
complex 2D numpy array (with any pre-processing
such as background removal already performed) ``hologram`` and a pre-computed ``propagator`` 
which can be generated using the function ``propagator``. Optionally specify ``imgIsFourier = True`` if ``hologram``
is provided as the FFT of the hologram (useful for speed up in some applications). GPU will be used if available
for faster refcousing, optionally specify ``cuda == False`` to prevent use of GPU.

.. py:function:: refocus_and_score(depth, imgFFT, pixelSize, wavelength, method, scoreROI, propLUT)

Used by find_foucs to refocus an image to specificed depth and returns focus score. ``depth``, ``pizelSize`` and ``wavelength`` are as defined
for ``refocus``, ``method`` is as defined for ``focus_score``. The focus scoring will be applied to
``scoreROI`` is an instance of ``ROI``, specify this as ``None`` to score the whole image. ``propLUT`` is
a pre-generated proagator look up table used to improve speed , specify as ``None`` to generate propagators on-the-fly instead. 
Returns the focus score. 

.. py:function:: refocus_stack(img, wavelength, pixelSize, depthRange, nDepths [, background = None, window = None] )

Numerical refocusing of a hologram to produce a depth stack. `'depthRange'` is a tuple
defining the min and max depths, the resulting stack will have ``nDepths`` images
equally spaced between these limits. Optionally specify a ``background`` and ``window``. 
Returns stack of refocused images as a 3D numpy array.


