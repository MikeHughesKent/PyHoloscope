^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Inline Holography Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative to using the :doc:`holo` as described in `Inline Holography Using the Holo Class <inline.html>`_, 
low-level functions can be called directly. For most applications this is
not the recommended approach as it involves more steps and the API is more likely to change in the future, however it
may be necessary if implementing more customised processing pipelines.

Begin by importing the library::
    
    import pyholoscope as pyh

Before we can refocus we define a propagator using :func:`pyholoscope.propagator`. This requires specification of the 
hologram size, wavelength, pixel size and the depth we wish to refocus to, e.g.::

    grid_size = 1024
    wavelength = 0.5e-6    # metres
    pixel_size = 2.0e-6    # metres
    depth = 1.0e-3         # metres
    prop = pyh.propagator(grid_size, wavelength, pixel_size, depth)

Assuming we have an inline hologram as a 2D numpy array ``hologram`` we can then refocus using :func:`pyholoscope.refocus`::

    refocused_img = pyh.refocus(hologram, propagator, background = background_img)

Here we have also provided an optional background hologram, ``background_img``, again a 2D numpy array. 
The returned image is a 2D complex numpy array, to obtain the amplitude image as 2D numpy array use::

    refocused_amp = pyh.amplitude(refocused_img)
    
Flat-fielding (normalisation) and windowing can also be applied by passing 2D numpy arrays using ``normalise=`` and ``window=`` respectively.
Windows can be generated manually or by using the :func:`pyholoscope.circ_window`, :func:`pyholoscope.circ_cosine_window`` or :func:`pyholoscope.square_cosine_window` functions. For example, to 
create a square cosine window which drops to 0 at the edges of the image, with a skin thickness of 10 pixels we could do the following::

    imgSize = np.shape(hologram)
    radius = np.shape(hologram) / 2
    skin_thickness = 10
    window = pyh.square_cosine_window(imgSize, radius, skin_thickness)
 
and then call::

    refocused_img = pyh.refocus(hologram, propagator, background = background_img, window = window, normalise = background_img)


    
    
    