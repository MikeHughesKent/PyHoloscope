----------
Holo Class
----------

The ``Holo`` class is the preferred way to access the core functionality of PyHoloscope. The class is generally used by setting various parameters, either by passing them at instantiation or by calling methods, and then by calling the ``process`` method to process each raw hologram.

Examples are provided of `Inline Holography <https://github.com/MikeHughesKent/PyHoloscope/tree/main/examples/inline_example.py>`_ and 
`Off-Axis Holography <https://github.com/MikeHughesKent/PyHoloscope/tree/main/examples/off_axis_example.py>`_ in the Github repository. Also see the `Inline Holography <inline.html>`_ 
and `Off-Axis Holography <off_axis.html>`_ getting started pages.


^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: Holo(mode, wavelength, pixelSize, optional arguments)
   :noindex:

Intantiation of a Holo object. ``mode`` determines the pipeline of processing to apply to images, either ``PyHoloscope.offaxis`` or ``PyHoloscope.inline``.
``wavelength`` is the light wavelength, ``pixelSize`` is the camera pixel size (as projected onto the object plane if magnification is present). 
``wavelength`` and ``pixelSize`` should be in the same units, and the same units must later be used for refocusing distance, if relevant.

There are a large number of additional optional keywords arguments, specified for example as ``useNumba = True``, which are listed below:


**General:**

* ``useNumba`` : boolean, ``True`` to use Numba JIT if available (default = ``True``).
* ``cuda`` : boolean, ``True`` to use CUDA for GPU processing if available (default = ``True``).
* ``precision`` : str, ``'single'`` or ``'double'``, determines output precision (default = ``'single'``).

**Background and normalisation**

These properties control background subtraction and normalisation (flat-fielding).

* ``background`` : 2D numpy array, background hologram. For off-axis holography, this is used to recover relative amplitude and relative phase if ``relativeAmplitude`` and ``relativePhase`` are True, respectively. For inline holography, the background hologram is subtracted prior to refoucsing, and if it is ``None`` there is no background subtraction (default = ``None``).
* ``normalise`` : 2D numpy array, normalisation image used for flat-field correction. If specified, this is used to produce a flat amplitude map prior to refocusing. (default = ``None``).

 
**Numerical refocusing:**

These properties control numerical refocusing using the angular spectrum method. Numerical refocusing is always performed when
performing inline holography, but must be specified by setting ``refocus = True`` for off-axis holography.

* ``refocus`` : boolean, if ``True`` and in off-axis mode, numerical refocusing will be performed (refocusing is **always** performed for inline mode, regardless of this setting) (default = ``False``).
* ``depth`` : float, distance to refocus to in same units as wavelength and pixelSize (default = 0).      
* ``downsample`` : float, hologram will be downsampled by this factor before refocusing for improved speed (default = 1, i.e. no downsampling).        


**Refocus Widowing:**

These properties control whether a spatial window is applied before and after numerical refocusing, and the properties of the window.

* ``autoWindow`` : boolean, if ``True``, a spatial cosine window will be created the first time a hologram is refocused, and applied *prior* to the numerical refocusing. The shape and size of the window are determined by the parameters below. (default = ``False``).
* ``postWindow`` : boolean, if ``True`` the window will also be applied *after* refocusing (default = ``False``).
* ``window`` : 2D numpy array or ``None``, a custom spatial window to use. This must be the same size as the raw hologram (for inline holography) or the demodulated hologram (for off-axis holography). (default = ``None``) 
* ``windowShape`` : str, the shape of the automatically generated window, ``'circle'`` or ``'square'`` (default = ``'square'``)        
* ``windowRadius`` : int or ``None``, radius of circular window or half-side length of sqaure window. If ``None``, the window will be the same size as the hologram. (default = ``None``)
* ``windowThickness`` : float, thickness of cosine smoothed edge of window in pixels (default = 10).

   
**Autofocus:**

These properties control the behaviour of the auto focusing methods.

* ``findFocusMethod`` : str, focus metric to use, (default = ``'Brenner'``)
* ``findFocusRoi``: Roi or None, region of interest to apply focus metric to (default = ``None``)
* ``findFocusMargin`` : int or None, margin around ``findFocusRoi`` to refocus. If ``None``, the whole image is refocused during focus search (default = ``None``).
* ``findFocusCoarseSearchInterval`` : int or None, if specified, a coarse focus search will be performed at a this number of positions (default = ``None``)
* ``findFocusDepthRange`` : (float, float), depth range to search between (default = (0,1))


**Off-axis demodulation:**

These properties control the behaviour of the demodulation step of off-axis holography.

* ``cropCentre`` : (int, int), location of modulation peak in FFT (default = ``None``)
* ``cropRadius`` : (int) or (int, int), semi-diameter of region to crop around modulation peak. Provide a single value for a square and a tuple of (w,h) for a rectange. (default = ``None``)
* ``returnFFT`` : boolean, if True ``process`` will return a tuple of the demodulated hologram and the FFT. (default = ``False``)
* ``cropMask`` : (default = ``None``)
* ``customCropWindow`` : 2D numpy array or None, a custom mask to apply to the cropped region. Must be same size as cropped region, as determined by ``cropRadius``. (default = ``None``)
* ``cropWindowSkinThickness`` : float or (float, float), if using a cosine window, this is the thickness of the smootheed edge. If a tuple if provided, the horizontal and vertical sides can have different thickness. (default = 10)


**Phase:**

These properties control the phase part of the returned complex reconstruction.

* ``relativeAmplitude`` : boolean, if ``True`` then for off-axis holography, if a 2D numpy array was provided in ``background``, the returned reconstruction will have an amplitude relative to the background hologram amplitude (default = ``False``).
* ``relativePhase`` : boolean, if ``True`` then for off-axis holography, if a 2D numpy array was provided in ``background``, the returned reconstruction will have a phase map which is relative to the background hologram phase map (default = ``False``).
* ``stablePhase`` : boolean, if ``True``, then for off-axis holography, if a ROI is provided in ``stableROI``, then the returned reconstruction will have a phase relative to the average phase in the ROI (default = ``False``).
* ``stableROI`` : Roi, a region of interest used for stable phase (default = ``None``)
 
 
**Display:**

* ``invert`` : boolean, if ``True``, output image is brightness inverted (default = ``False``)



^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^


.. py:function::  apply_window(img)

Apply the current window to a hologram ``img`` and return the windowed hologram as a 2D numpy array.
      
   

.. py:function:: auto_find_off_axis_mod()

Detect and store the modulation parameters for off-axis demodulation. 


 
.. py:function:: auto_focus(img, [optional arguments]):

A more heavily customisable auto-focus, for most purposes used ``find_focus`` instead.



.. py:function:: calib_off_axis(hologram)   

Determines the modulation frequency and crop radius for off-axis holgraphy using ``hologram``, a 2D numpy array, and stores the results
internally for when ``process()`` is called. 


     
.. py:function:: clear_background()

Remove a previously set background, equivalent to calling ``set_background(None)``.


        
.. py:function::  clear_propagator_LUT()

Deletes a previously created look up table (LUT) of propagators.
  


.. py:function::  depth_stack(img, depthRange, nDepths)

Create a depth stack of refocused images from a hologram ``img`` (either an inline hologram or a demodulated off-axis hologram) using current parameters, producing a set of ``nDepths`` refocused images. ``depthRange`` is a tuple of (min depth, max depth). Returns an instance of the class ``RefocusStack``. To refocus to this depth, set this as the new depth using ``set_depth``.
        

        
.. py:function::  find_focus(img):    

Automatically finds the best focus position for hologram ``img`` using parameters defined using ``set_find_focus_parameters``. Returns the depth of best focus.



.. py:function:: make_propagator_LUT(depthRange, nDepths)

Creates a LUT of propagators for faster finding of focus of a range of depths. ``nDepths`` is the number of propagators to generate, and ``depthRange`` is a tuple defining the minimum and
maximum depths to generate for.


 
.. py:function:: off_axis_background_field()

Performs off-axis demodulation of a background hologram which has been provided via ``set_background``.


   
.. py:function:: process(img)

Process an image ``img`` using the currently selected options. Returns the processed image as 2D complex Numpy array.


.. py:function:: set_auto_window(autoWindow)

Sets whether a window will be created and applied prior to refocusing
if one has not been specified. ``autoWindow`` is a Boolean.



.. py:function:: set_background(background)

Set the background image. Pass ``None`` to remove an existing background.



.. py:function:: set_depth(depth)

Set the depth for numerical refocusing. ``depth`` should be in the same units as ``wavelength`` and ``pixelSize``.

  
     
.. py:function:: set_downsample(downsample)

Set the downsampling factor. The holograms will be spatially downsampled by this factor. This will cause the propagator to be recreated when next needed, call ``update_propagator`` to force this immediately.



               
.. py:function:: set_find_focus_parameters([method = 'Brenner', depthRange = (0, 0.1), roi = None, margin = None, coarseSearchInterval = None ])

Sets the parameters used by the find_focus method. See `automatic depth determination <autofocus.html>`_ for details.
 


.. py:function:: set_oa_centre(centre)

Set the location of the modulation frequency in frequency domain. ``centre`` is is a tuple
of the (x,y) location of the modulation peak in the FFT of the hologram.

  
     
.. py:function:: set_oa_radius(radius)

Set the size of the region to extract in frequency domain for off-axis demodulation. ``radius`` is half the length of the side of a square
around the modulation peak in the FFT of the hologram.

    
    
.. py:function:: set_off_axis_mod(cropCentre, cropRadius)

Sets the location of the frequency domain position for off-axis modulation. ``cropCentre`` is a tuple
of the location of the modulation peak in the FFT of the hologram, ``cropRadius`` is the half the side length of
a square around the modulation peak that will be used to generate the demodulated image.


.. py:function:: set_precision(precision)

Sets the numerical precision to use internally, either 'single' (defualt) or 'double'.



.. py:function:: set_pixel_size(pixelSize)

Set the physical size of pixels in the raw hologram

     
     
.. py:function:: set_return_FFT(returnFFT)

If returnFFT is ``True`` the FFT rather than the reconstructed image will be returned when performing off-axis holography. 


  
.. py:function:: set_stable_ROI(roi)

Sets the location of the ROI used for maintaining a constant background phase, i.e. this should be a background region of the image. ``roi``
should be an instance of the ``Roi`` class.
   



.. py:function:: set_use_cuda(useCuda)

Sets whether to use GPU if available, pass ``true`` to use GPU (default) or ``false`` to not use GPU.


        
.. py:function:: set_use_numba(useNumba)

Sets whether to use Numba JIT if available, pass ``true`` to use Numba if available (defuault) or ``false`` to not use Numba.


.. py:function:: set_wavelength(wavelength)
    
Set the wavelength of the hologram.
     
 
.. py:function:: set_window(img, radius, thickness, [shape = 'square'])

Sets a cosine window used for pre and post processing to reduce edge artefacts.
``img`` is a 2D numpy array which is either the hologram or any numpy array of the same size as the hologram, 
``radius`` is the the size of the window
and ``thickess`` determines the distance over which the window tapers from 0 to 1.  By defualt the window is square, pass ``shape = 'circle'`` to generate a circular window.

 
.. py:function:: set_window_radius(windowRadius)

Sets the radius of the cropping window.
     
          
     
.. py:function:: set_window_thickness(windowThickness)

Sets the edge thickness of the cropping window.
    
         
.. py:function:: update_propagator(img)

Create or re-create the angular spectrum propagator using current parameters.
 

                    
        

    
    