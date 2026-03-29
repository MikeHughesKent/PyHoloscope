----------------------------------
Automatic Focusing
----------------------------------

The optimal refocus depth for a hologram can be determined using trial refocuses, assessed using a focus metric. 
The package supports an exhaustive search or a much faster search using Brent's method. The focus can be
optimised for the entire image or for a region of interest (ROI). Faster determination for a ROI can be achieved
by only refocusing a subset of the image consisting of the ROI and a margin around it. It is also possible to pre-compute
a look up table of propagators to provide a further speed-up.

If using the ``Holo`` class then the focus can be found and the image refocused using the :func:`pyholoscope.Holo.auto_focus` method, or the :func:`pyholoscope.Holo.find_focus` method can be used
to obtain the focus depth without actually refocusing. Alternatively, the lower level :func:`pyholoscope.find_focus` function can be used directly.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Holo class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh

For refocusing we need to define the image pixel size, and wavelength, for example::

    wavelength = 520e-9
    pixel_size = 4.8e-6 
    
We also define the refocus depth range to search over, and the focus metric (method) to apply::

    depth_range = (.08, .16)    
    method = 'sum'              
  
We will assume we have a hologram ``hologram`` stored as a 2D numpy array. This can either be a raw inline hologram, in 
which case ``hologram`` will be real, or a demodulated off-axis hologram, in which case ``hologram`` will be complex.
    
We choose a region of interest (ROI) around the interesting part of the image::

    roi = pyh.Roi(450,110,500,400)

We then create the Holo object, optionally specifying a background::

    holo = pyh.Holo(mode = pyh.INLINE, wavelength = wavelength, pixel_size = pixel_size, background = background)

We set the focus parameters, optionally specifying the region of interest::

    holo.set_find_focus_parameters(depth_range = depth_range, roi = roi, method = method)

We then use :func:`pyholoscope.Holo.auto_focus` to obtain a refocused image at the best focal depth::

    refocused = holo.auto_focus(hologram)

This also sets ``holo.depth`` to the best focal depth. If we just wanted to find
the best focal depth without producing a refocused image, we can call :func:`pyholoscope.Holo.find_focus`::

    focus_depth = holo.find_focus(hologram)
 
If we only want to refocus a margin around the region of interest when searching for the best focus, 
for improved speed, we can specify the ``margin``, for example::

    holo.set_find_focus_parameters(depth_range = depth_range, roi = roi, method = method, margin = 20)

For further improved speed we can pre-compute a propagator LUT by deciding how many propagator depths
we would like to pre-calculate, and then calling::

    num_depths = 100
    holo.make_auto_focus_propagator_LUT(hologram, depth_range, num_depths, roi=roi, margin=margin)
    
where we must provide the correct ``roi`` and ``margin`` if a margin is to be used, to ensure the
propagator LUT has propagators of the correct size. We then indicate we would like to use the LUT::
    
    holo.set_find_focus_parameters(depth_range = depth_range, roi = roi, method = method, margin = margin, use_prop_lut = True)

Now, we call ``auto_focus`` or ``find_focus`` as before. ``num_depths`` must be chosen to 
give the required focus precision, which will depend on the imaging system and application.
    

^^^^^^^^^^^^^
Focus Metrics
^^^^^^^^^^^^^

The available options for the focus metric are below. A list of strings of the available methods can be obtained from :func:`pyholoscope.get_focus_score_methods`.

- ``sum`` - Takes the sum of all pixel values, for amplitude images this will be lowest when the image is in focus. 
- ``peak`` - The largest single pixel value, this is usually highest for in focus images.
- ``brenner`` - Applies the Brenner edge detection filter across the image and takes the mean.       
- ``sobel`` - Applies a Sobel edge detection filter and takes the mean
- ``sobel_variance`` - Applies a Sobel edge detection filter and takes the standard deviation
- ``dark_focus`` - Takes the standard deviation of the quadrature sum of the vertical and horizontal image gradients
- ``norm_var`` - Normalised variance.

Where appropriate the output from the metrics are negated to ensure the best focus also has the lowest score.

Custom metrics can be defined as functions, with the function passed to the auto/find focus functions instead of a string. Custom
metrics must accept a single argument, the hologram, and return a single focus score, with lower values meaning better focus.    
    

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
We will assume we have a hologram ``hologram`` stored as a 2D numpy array. This can either be a raw inline hologram, in 
which case ``hologram`` will be real, or a demodulated off-axis hologram, in which case ``hologram`` will be complex.
    
For refocusing we need to define the image pixel size, and wavelength, for example::

    pixel_size = 2e-6
    wavelength = 0.5e-6
    
We also need to define the range of depths to search over as a tuple of ``(minDepth, maxDepth)``, and select
the focus metric to use, for this example we will choose the 'sum' metric::
    
    depth_range = (0, 0.001)
    method = 'sum'
    
We then call :func:`pyholoscope.find_focus`::
    
    focusDepth = pyh.find_focus(hologram, wavelength, pixel_size, depth_range, method)
   
This returns the best refocus depth, found using Brent's method.

If we would like to find the best focus for a specific region of the image only, we define a ROI such as::

    x = 20
    y = 30
    w = 40
    h = 40
    focus_roi = pyh.Roi(x,y,w,h)
    
and pass this as an optional argument::

    focus_depth = pyh.find_focus(hologram, wavelength, pixel_size, depth_range, method, roi = focus_roi)
    
While the focus metric will be applied only to the ROI, the whole hologram is refocused to each trial depth, so there is usually no significant speed-up 
from doing this.

A speed-up can be obtained by refocusing only the ROI plus a small margin around it. This is activated by specifying the margin in pixels::

    focus_depth = pyh.find_focus(hologram, wavelength, pixel_size, depth_range, method, roi = focus_roi, margin = 20)

If the hologram has not had a background subtracted or a window previously applied, we can also request these are applied prior to refocusing by 
passing them as optional arguments::

    window = pyh.circ_cosine_window(imageSize, imageSize / 2, 20)
    focus_depth = pyh.find_focus(hologram, wavelength, pixel_size, depth_range, method, roi = focus_roi, margin = 20, background = background_img, window = window)      


To use a propagator LUT, this is first created using :class:`pyholoscope.PropLUT`::

    nDepths = 100
    lut = pyh.PropLUT(imgSize, wavelength, pixel_size, depth_range, nDepths)
    
and then passed to the ``find_focus`` function::

    focusDepth = pyh.find_focus(hologram, wavelength, pixel_size, depth_range, method, prop_lut = lut)      

If also attempting to speed-up by refocusing only a ROI (i.e. the margin is specified) then it is necessary to create a propagator LUT of the correct size for 
this ROI + margin. There is a helper function for this: :func:`pyholoscope.propagator_size_for_roi`::

    prop_size = pyh.propagator_size_for_roi(hologram, roi = roi, margin = margin)
    prop_lut = pyh.PropLUT(prop_size, wavelength, pixel_size, depth_range, num_depths)
