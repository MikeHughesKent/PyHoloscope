--------------------------------
Inline Holography Basics
--------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using OOP
^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
And instantiate an object. At this point we need to provide the image size, pixel size, wavelength, and initialise for inline holography::

    imageSize = 512
    pixelSize = 2e-6
    wavelength = 0.5e-6
    holo = pyh.Holo(pyh.INLINE_MODE, imageSize, pixelSize, wavelength)
    
Inline holography requires a background image for good quality reconstructions. Assuming the 
background image is stored in the 2D numpy array ``backgroundImg``, use::

    holo.set_background(backgroundImg)
    
We can now numerically refocus a hologram ``hologram``, again a 2D numpy array::

    refocusedImg = holo.process(hologram)

The output, ``refocusedImg``, is complex, we can obtain the amplitude using::

    refocusedAmp = pyh.amplitude(refocusedIm)
    
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the library::
    
    import pyholoscope as pyh

Before we can refocus we define a propagator. This requires specification of the hologram size, wavelength, pixel size and the depth we wish to refocus to::

    gridSize = 1024
    wavelength = 0.5e-6
    pixelSize = 2e-6
    depth = 1e-3
    prop = pyh.propagator(gridSize, wavelength, pixelSize, depth)

Assuming we have an inline hologram ``hologram`` we can then refocus using::

    refocusedImg = pyh.refocus(hologram, propagator, background = backgroundImg)

Here we have also provided an optional background hologram, ``backgroundImg``. The returned image is complex, to obtain the amplitude we use::

    refocusedAmp = pyh.amplitude(refocusedAmp)
    
        
^^^^^^^^^^^^^^^^
GPU acceleration
^^^^^^^^^^^^^^^^
GPU acceleration is used by default when using OOP, it can be enabled/disabled using::

    holo.set_cuda(True/False)

This requires the CuPy package and a compatible GPU, otherwise pyholoscope will revert to CPU processing.  

If using the lower level functions, specify ``cuda = True`` when refocusing, e.g. ::

    holo.refocus(hologram, propagator, cuda = True)

    
    
    