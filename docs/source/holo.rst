----------
Holo Class
----------

The ``Holo`` class is the preferred way to access the core functionality of PyHoloscope. A full list of method is below. Examples are provided in 'examples\\inline_examply.py' and 'examples\\off_axis_example.py' in the Github repository. Also see the `Inline Holography <inline.html>`_ 
and `Off-Axis Holography <off_axis.html>`_ getting started pages. The class is generally used by setting various parameters and then calling ``process`` to process individual holograms.

^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: Holo(mode, wavelength, pixelSize, **kwargs)

Intantiation of a Holo object. ``mode`` determines the pipeline of processing to apply to images, either ``PyHoloscope.offaxis`` or ``PyHoloscope.inline``.
``wavelength`` is the light wavelength, ``pixelSize`` is the camera pixel size (as projected onto the object plane if magnification is present). 
``wavelength`` and ``pixelSize`` should be in the same units. Returns an instance of ``Holo`` class.

^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^


.. py:function::  apply_window(img)

Apply the current window to a hologram ``img``.        

   

.. py:function:: auto_find_off_axis_mod()

Detect and store the modulation parameters for off-axis demodulation. 



 
.. py:function:: auto_focus(img, [optional arguments]):

A more heavily customisable auto-focus, for most purposes used ``find_focus`` instead.


     
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

Process an image ``img`` using the currently selected options. RTeturns the processed image as 2D complex Numpy array.






.. py:function:: set_background(background)

Set the background image. Pass ``None`` to remove an existing background.



.. py:function:: set_depth(depth)

Set the depth for numerical refocusing. ``depth`` should be in the same units as ``wavelength`` and ``pixelSize``.

  
     
.. py:function:: set_downsample(downsample)

Set the downsampling factor. The holograms will be spatially downsampled by this factor. This will cause the propagator to be recreated when next needed, call ``update_propagator`` to force this immediately.



               
.. py:function:: set_find_focus_parameters([method = 'Brenner', depthRange = (0, 0.1), roi = None, margin = None, coarseSearchInterval = None ])

Sets the parameters used by the find_focus method. See ``automatic depth determination <autofocus.html>`_ for details.
 


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
     
 
.. py:function:: set_window(img, radius, thickness, [shape = 'circle'])

Sets a cosine window used for pre and post processing to reduce edge artefacts.
`img`` is a 2D numpy array which is either the hologram or any numpy array of the same size as the hologram, 
``radius`` is the the size of the window
and ``thickess`` determines the distance over which the window tapers from 0 to 1.  By defualt the window is circular, pass ``shape = 'square'`` to generate a square window.

 
.. py:function:: set_window_radius(windowRadius)

Sets the radius of the cropping window.
     
     
     
     
.. py:function:: set_window_thickness(windowThickness)

Sets the edge thickness of the cropping window.

    
         
.. py:function:: update_propagator(img)

Create or re-create the angular spectrum propagator using current parameters.
 

                    
        

    
    