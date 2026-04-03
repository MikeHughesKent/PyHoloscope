----------------------------------
Inline Holography Using Holo Class
----------------------------------
Inline Holography can be performed using the :doc:`holo` of PyHoloscope by setting ``mode = pyholoscope.INLINE``. This allows
numerical refocusing using the angular spectrum or Fresnel propagator methods, as well as optional background subtraction, normalisation, windowing and autofocusing.
See the `Holo class documentation <holo.html>`_ for a full list of methods and arguments. For code examples see the `Inline Holography Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/inline_example.py>`_
or `Inline Holography Advanced Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/inline_example_advanced.py>`_ on github.

Alternatively, for more customisability, the low-level functions can be used directly as described in 
`Inline Holography Using Lower-Level Functions <inline_low_level.html>`_ .

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using the Holo class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the PyHoloscope package::

    import pyholoscope as pyh
    
and create an instance of the :doc:`holo`. At a minimum we need to set the mode to inline
holography and provide the physical pixel size and the wavelength::

    holo = pyh.Holo(mode = pyh.INLINE, pixel_size = 2e-6, wavelength = 0.5e-6)
    
The pixel size and wavelength should be specified in the same units (e.g. metres). Subsequently, the refocus depth 
should be specified in the same units.
    
Better quality inline holography refocusing can be achieved if we first
subtract a background image, acquired with no object in the field-of-view, forming
what is known as a contrast hologram. Assuming the background image is stored in the 2D numpy array ``background_img``, 
a background can be specified using::

    holo.background = background_img
    
or by passing ``background = background_img`` as an argument when creating the ``Holo`` object:: 

    holo = pyh.Holo(mode = pyh.INLINE, pixel_size = 2e-6, wavelength = 0.5e-6, background = background_img)

If we would like to divide through by a flat-field image stored as a 2D numpy array ``flat_img``, to correct for
intensity variations, we can pass ``normalise = flat_img`` or call
``holo.normalise = flat_img``.

We can now numerically refocus a hologram ``hologram``, again a 2D numpy array, 
using the angular spectrum method by first setting the depth to refocus to, for example::
 
    holo.depth = 0.005

(or by passing ``depth = 0.005`` when instantiating ``Holo``) and then calling::

    refocused_img = holo.process(hologram)

The output, ``refocused_img``, is a 2D complex numpy array; we can obtain the amplitude as a 2D float numpy array using::

    refocused_amp = pyh.amp(refocused_img)
    
Note that the first time a hologram is refocused to a particular depth the process 
will be slower due to the need to create a propagator for that depth. This is 
particularly noticeable when using GPU acceleration as the propagator creation 
will often be the rate-limiting step. Subsequent refocusing to the same depth 
will be faster providing no parameters are changed that force a new propagator 
to be created (depth, pixel size, wavelength or grid size). 
    
The angular spectrum propagator and the window are both created the first time
``process`` is called for a particular set of parameters. If we prefer to pre-generate these, we can call::

    holo.update_propagator(img)
    holo.update_auto_window(img)

where ``img`` is a 2D numpy array of the size of the hologram to be processed.


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Windowing
^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

If we would like to smooth the edges of the hologram, we can apply a window before
refocusing by calling:: 

    holo.auto_window = True
   
By default the window will be a rectangular cosine window. Options for the window size and shape
are set using the ``window_shape``, ``window_radius`` and ``window_thickness`` attributes
of :doc:`holo`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Propagation Models and Geometric Magnification Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
If required, the wavefront propagation model can be switched by changing ``propagation_method`` from ``"angular_spectrum"`` (default) to ``"fresnel"``.
For example::

    holo.propagation_method = "fresnel"

While less accurate than the angular spectrum method in some cases, the Fresnel method is slightly faster to 
compute and may be sufficient for many applications.

When using a point source, the effective pixel size is not the physical pixel camera size,
as the hologram is magnified onto the camera due to projection effects. The ``depth`` value
for which focus is achieved will therefore not be the true distane to the sample plane.

Effective-magnification correction can be enabled by setting ``correct_pixel_size = True`` and
providing ``source_distance`` (source-to-camera distance), for example::

    holo = pyh.Holo(mode = pyh.INLINE, pixel_size = 2e-6, wavelength = 0.5e-6, depth = 0.005,
                    source_distance = 0.01, correct_pixel_size = True)


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Numba JIT acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
If the Numba package is installed, this will be employed for faster generation 
of propagators by default when using the ``Holo`` class. Use of Numba can be 
explicitly enabled/disabled using:: 
        
    holo.use_numba = True/False
    
   

^^^^^^^^^^^^^^^^
GPU acceleration
^^^^^^^^^^^^^^^^
GPU acceleration is used by default when using the ``Holo`` class, it can be 
explicitly enabled/disabled using::

    holo.cuda = True/False

This requires the CuPy package and a compatible GPU, otherwise ``Holo`` will 
revert to CPU processing.  


