--------------------------------
Inline Holography Basics
--------------------------------
Inline Holography is performed using the ``Holo`` class of PyHoloscope by setting ``mode = pyholoscope.INLINE``. This allows
numerical refocusing using the angular spectrum method, as well as optional backgroud subtraction, normalisation and windowing.
See the `Holo class documentation <holo.html>`_ for a full list of methods and arguments. For code examples see the `Inline Holography Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/inline_example.py>`_
`Inline Holography Advanced Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/inline_example_advanced.py>`_ on github.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using the Holo class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the PyHoloscope package::

    import pyholoscope as pyh
    
and create an instance of the ``Holo`` class. At a minimum we need to set the mode to inline
holography and provide the physical pixel size and the wavelength::

    holo = pyh.Holo(mode = pyh.INLINE, pixelSize = 2e-6, wavelength = 0.5e-6)
    
The pixel size and wavelength can be in any units as long as they are the same, 
and subsequently the refocus depth will be in the same units.
    
Better quality inline holography refocusing is normally achieved if we first
subtract a background image, acquired with no object in the field-of-view.
Assuming the background image is stored in the 2D numpy array ``backgroundImg``, 
a background can be specified using::

    holo.set_background(backgroundImg)
    
or by passing ``background = backgroundImg`` as an argument when creating the ``Holo`` object. 
If we would like to divide through by a background image, to correct for
intensity variations, we can pass ``normalise = backgroundImg`` or call
``set_normalise(backgroundImg)``.

We can now numerically refocus a hologram ``hologram``, again a 2D numpy array, 
using the angular spectrum method by first setting the depth to refocus to, for example::
 
    holo.set_depth(0.005)

(or by passing ``depth = 0.005`` when creating holo) and then calling::

    refocusedImg = holo.process(hologram)

The output, ``refocusedImg``, is a 2D complex numpy array; we can obtain the amplitude as a 2D float numpy array using::

    refocusedAmp = pyh.amplitude(refocusedIm)
    
Note that the first time a hologram is refocused to a particular depth the process 
will be slower due to the need to create a propagator for that depth. This is 
particularly noticable when using GPU acceleration as the propagator creation 
will often be the rate limiting step. Subsequent refocusing to the same depth 
will be faster providing no parameters are changed that force a new propagator 
to be created (depth, pixel size, wavelength or grid size). 

If we would like to smooth the edges of the hologram, we can apply a window before
refocusing by calling:: 

    holo.set_auto_window(True)
 
The angular spectrum propagator and the window are both created the first time
``process`` is called. If we prefer to pre-generate these, we can call::

    holo.update_propagator()
    holo_update_auto_window()
    


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Numba JIT acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
If the Numba package is installed, this will be employed for faster generation 
of propagators by default when using the ``Holo`` class. Use of Numba can be 
explicitly enabled/disabled using:: 
        
    holo.set_numba(True/False)
    
If using the lower-level functions, the Numba variant of the propagator generator function must be called explicitly::

    pyh.propagator_numba(gridSize, wavelength, pixelSize, depth)    
    

^^^^^^^^^^^^^^^^
GPU acceleration
^^^^^^^^^^^^^^^^
GPU acceleration is used by default when using the ``Holo`` class, it can be 
explictly enabled/disabled using::

    holo.set_cuda(True/False)

This requires the CuPy package and a compatible GPU, otherwise ``Holo`` will 
revert to CPU processing.  

If using the lower-level functions, it is necessary to specify ``cuda = True``
when refocusing, e.g.::

    holo.refocus(hologram, propagator, cuda = True)

    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative to using the ``Holo`` class, low-level functions can be called directly. Begin by importing the library::
    
    import pyholoscope as pyh

Before we can refocus we define a propagator. This requires specification of the hologram size, wavelength, pixel size and the depth we wish to refocus to::

    gridSize = 1024
    wavelength = 0.5e-6
    pixelSize = 2e-6
    depth = 1e-3
    prop = pyh.propagator(gridSize, wavelength, pixelSize, depth)

Assuming we have an inline hologram as a 2D numpy array ``hologram`` we can then refocus using::

    refocusedImg = pyh.refocus(hologram, propagator, background = backgroundImg)

Here we have also provided an optional background hologram, ``backgroundImg``, again a 2D numpy array. 
The returned image is a 2D complex numpy array, to obtain the amplitude image as 2D numpy array use::

    refocusedAmp = pyh.amplitude(refocusedAmp)
    
        
    
    