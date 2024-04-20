----------------------------------
Off-Axis Holography Basics
----------------------------------
Off-axis Holography is performed using the ``Holo`` class of PyHoloscope by setting ``mode = pyholoscope.OFFAXIS``. This allows
demodulation of the off-axis carrier frequency to recover the quantitative phase as well numerical refocusing. See the 
`Holo class documentation <holo.html>`_  for a full list of methods and arguments. For code examples see the `Off-axis Holography Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/off_axis_example.py>`_ and the
`Off-axis Holography with Refocusing Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/off_axis_refocus_example.py>`_ on github.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using Holo Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
and create an instance of the ``Holo`` class. At a minimum we need to set the mode to off-axis
holography and, if we would like to perform numerical refocusing, provide the physical pixel size and the wavelength::

    holo = pyh.Holo(mode = pyh.OFF_AXIS, pixelSize = 2e-6, wavelength = 0.5e-6)
    
We also need to know the spatial frequency of the modulation. We can determine this automtically using::

    holo.calib_off_axis(backgroundImg)         
    
where we have provided a background hologram to use for this purpose. It is possible 
to use the image hologram as well for this purpose, but this may be unreliable if there
is another strong spatial frequency.  If a background image has first been set using::

    holo.set_background(backgroundImg)
    
or by passing ``background = backgroundImage`` when creating the ``Holo`` object, then
this will be used for the calibration if ``calib_off_axis`` is called with no argument.

Alternatively the demodulation parameters can be specified manually using::

    holo.set_off_axis_mod(cropCentre, cropRadius)
   
where ``cropCentre`` is a tuple of (x,y), giving the pixel location of the centre of the modulation peak in the FFT of the hologram, 
and ``cropRadius`` is half the size of the box around the modulation centre which is demodulated.  ``cropRadius`` can also
be a tuple giving the x and y radii of an ellipse, for cases where the hologram is non-square.  

We can then demodulate to obtain the complex field using::

    reconField = hol.process()
    
To obtain the amplitude and phase, use::

    amplitude = pyh.amplitude(reconField)
    phase = pyh.phase(reconField) 

If we would like to also refocus to a different depth we can specify this when we create the ``Holo`` object::

    holo = pyh.Holo(mode = pyh.OFFAXIS_MODE, pixelSize = 2e-6, wavelength = 0.5e-6, refocus = True, depth = 0.001)
        
Then when we call::

    reconField = holo.process()
    
Both the demodulation and the refocusing will take place in a single step.
    
We can change the refocus depth and whether or not to refocus witout recreating the ``Holo`` object using::

    holo.set_depth(depth)
    holo.set_refocus(True)    

Note that the first time a hologram is refocused to a particular depth the process will be slower 
due to the need to create a propagator for that depth. This is particularly noticable when using
GPU acceleration as the propagator creation will often be the rate limiting step. Subsequent 
refocusing to the same depth will be faster providing no parameters are changed that force 
a new propagator to be created (depth, pixel size, wavelength or grid size). The propagator can
also be pre-computed by calling::

    holo.update_propagator()
 
in advance.

To correct for a background phase (i.e the phase map of the background hologram), set::

    holo.set_relative_phase(True)
    
or pass ``relative_phase = True`` when creating the ``Holo`` object. You should then call::

    holo.background_field()
    
to compute the background phase.   
  
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative to using the ``Holo`` class, low-level functions can be called directly. Begin by importing the library::
    
    import pyholoscope as pyh
    
We will assume we have a hologram ``hologram`` and a background image ``backgroundImg`` which are both square 2D numpy arrays of the same size. 
If we do not know the modulation frequency in advance we can use::

    cropCentre = pyh.off_axis_find_mod(backgroundImg)
    cropRadius = pyh.off_axis_find_crop_radius(backgroundImg)  
    
We can then demodulate using::

    reconField = pyh.off_axis_demod(hologram, cropCentre, cropRadius)
    
To remove the background, recover the background field using::

    backgroundField = pyh.off_axis_demod(background, cropCentre, cropRadius)  
    
Remove the background phase (for example to due to aberrations in the imaging system) using::

    correctedField = pyh.relative_phase(reconField, backgroundField)
    
The numpy array ``correctedField`` is complex, to obtain the amplitude and phase, use::

    amplitude = pyh.amplitude(reconField)
    phase = pyh.phase(reconField) 
  
If we would like to numerically refocus, we first define a propagator for use with the angular spectrum method. 
This requires specification of the hologram size, wavelength, pixel size and the depth we wish to refocus to::

    gridSize = cropRadius * 2
    wavelength = 0.5e-6
    pixelSize = 2e-6
    depth = 1e-3
    prop = pyh.propagator(gridSize, wavelength, pixelSize, depth)
    
Note here that the ``gridSize`` is the size of the reconstructed field following demodulation which is smaller than the original image. 
The pixel size must also be specified as the pixel size in the reconstructed field, not the pixel size in the original hologram. 
Pixel size, wavelength and depth must be in the same units.
 
We can then refocus using::

    refocusedImg = pyh.refocus(correctedField, propagator)

The numpy array ``refocusedField`` is a 2D complex numpy array, to obtain the amplitude and phase as 2D numpy arrays, use::

      amplitude = pyh.amplitude(refocusedField)
      phase = pyh.phase(refocusedField)


