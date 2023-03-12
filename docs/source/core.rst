--------------------------------
Inline Holography : Introduction
--------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using OOP
^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import PyHoloscope as holo
    
And instantiate an object. At this point we need to provide the image size, pixel size, wavelength::

    imageSize = 512
    pixelSize = 2e-6
    wavelength = 0.5e-6
    hol = PyHoloscope.Holo(imageSize, pixelSize, wavelength)
    
Inline holography requires a background image for good quality reconstructions. Assuming the 
background image is stored in the 2D numpy array ``backgroundImg``, use::

    hol.set_background(backgroundImg)
    
We can now numerically refocus a hologram ``hologram``, again a 2D numpy array::

    refocusedImg = hol.process(hologram)

The output, ``refocusedImg``, is complex, we can obtain the amplitude using::

    refocusedAmp = holo.amplitude(refocusedIm)

    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the library::
    
    import PyHoloscope as holo

Before we can refocus we define a propagator. This requires specification of the hologram size, wavelength, pixel size and the depth we wish to refocus to::

    gridSize = 1024
    wavelength = 0.5e-6
    pixelSize = 2e-6
    depth = 1e-3
    prop = holo.propagator(gridSize, wavelength, pixelSize, depth)

Assuming we have an inline hologram ``hologram`` we can then refocus using::

    refocusedImg = holo.refocus(hologram, propagator, background = backgroundImg)

Here we have also provided an optional background hologram, ``backgroundImg``. The returned image is complex, to obtain the amplitude we use::

    refocusedAmp = holo.amplitude(refocusedAmp)
    
        
^^^^^^^^^^^^^^^^
GPU acceleration
^^^^^^^^^^^^^^^^
To enable GPU acceleration for refocusing using OOP, use::

    hol.set_cuda(True)

This requires the CuPy package and a compatible GPU.  

If using the lower level functions, specify ``cuda = True`` when refocusing, e.g. ::

    holo.refocus(hologram, propagator, cuda = True)

    
    
    