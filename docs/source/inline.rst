--------------------------------
Inline Holography Basics
--------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using OOP (Holo class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
and instantiate a ``Holo`` object. As a minimum we need to set the mode to inline
holography and provide the physical pixel size and the wavelength. The pixel size 
and wavelength should be specified in the same units, and subsequently the refocus 
depth must be in the same units.::

    holo = pyh.Holo(mode = pyh.INLINE, pixelSize = 2e-6, wavelength = 0.5e-6)
    
Inline holography benefits from a background image, acquired with no object in 
the field-of-view, for good quality reconstructions. 
Assuming the background image is stored in the 2D numpy array ``backgroundImg``, use::

    holo.set_background(backgroundImg)
    
or pass ``background = backgroundImg`` as an argument when creating the ``Holo`` object. 
We can now numerically refocus a hologram ``hologram``, again a 2D numpy array, 
using the angular spectrum method by first setting the depth to refocus to, for example::
 
    holo.set_depth(0.005)

(or by passing ``depth = 0.005`` when creating holo) and then calling::

    refocusedImg = holo.process(hologram)

The output, ``refocusedImg``, is a 2D complex numpy array, we can obtain the amplitude as a 2D float numpy array using::

    refocusedAmp = pyh.amplitude(refocusedIm)
    
Note that the first time a hologram is refocused to a particular depth the process will be slower due to the need to create a propagator for that 
depth. This is particularly noticable when using GPU acceleration as the propagator creation will often be the rate limiting step. 
Subsequent refocusing to the same depth will be faster providing no parameters are changed that force a new propagator to be created (depth, pixel size, wavelength or grid size). 

If we would like to smooth the edges of the hologram, we can apply a window before
refocusing by calling:: 

    holo.set_auto_window(True)
 
The angular spectrum propagator and the window are both created the first time
``process`` is called. If you would prefer to pre-generate these, you can call

    holo.update_propagator()
    holo_update_auto_window()


For a minimal example see 'examples/inline_example.py' in the Github repository.

For a more detailed example see 'examples/inline_example_advanced.py' in the Github repository.


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
    
        
    
    