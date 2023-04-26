----------------------------------
Off-Axis Holography Basics
----------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using OOP (Holo Class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
And instantiate a ``Holo`` object. We need to provide the input hologram image size in pixels (the hologram must be square), 
the physical pixels size and the wavelength. The pixel size and wavelength should be specified in the same units, 
and subsequently the refocu depth must be in the same units.::

    imageSize = 512
    pixelSize = 2e-6
    wavelength = 0.5e-6
    holo = pyh.Holo(pyh.OFFAXIS_MODE, imageSize, pixelSize, wavelength)
    
Off-axis holography benefits from a background image for good quality phase recovery. Assuming the 
background image is stored in the 2D numpy array ``backgroundImg``, use::

    holo.set_background(backgroundImg)
    
We also need to know the spatial frequency of the modulation. We can determine this automtically using::

    holo.auto_find_off_axis_mod(backgroundImg)         
    
It is possible to use the image hologram as well for this purpose, but this may be unreliable if there
is another strong spatial frequency. Alternatively it can be specified manually using::

    holo.set_off_axis_mod(cropCentre, cropRadius)
   
where ``cropCentre`` is a tuple of (x,y), giving the pixel location of the centre of the modulation peak in the FFT of the hologram, 
and ``cropRadius`` is half the size of the box around the modulation centre which is demodulated.    

We can then demoodulate to obtain the complex field at focus using::

    reconField = hol.process()
    
To obtain the amplitude and phase, use::

    amplitude = pyh.amplitude(reconField)
    phase = pyh.phase(reconField) 

If we would like to also refocus to a different depth we can specify::

    holo.set_refocus = True
    depth = 0.001
    holo.set_depth(depth)
        
Then when we call::

    reconField = holo.process()

Both the demodulation and the refocusing will take place in a single step.

Note that the first time a hologram is refocused to a particular depth the process will be slower due to the need to create a propagator for that 
depth. This is particularly noticable when using GPU acceleration as the propagator creation will often be the rate limiting step. 
Subsequent refocusing to the same depth will be faster providing no parameters are changed that force a new propagator to be created (depth, pixel size, wavelength or grid size). 


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Manually specifying the modulation frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If we know the spatial frequency in advance, rather than using ``auto_find_off_axis_mod`` we can set
the demodulation parameters manually. We need to specify the spatial frequency of the modulation frequency
in terms of the pixel location of the peak corresponding to the modulation in the 2D Fourier transform of the hologram. For example::
    
    centre = (200,220)
    holo.set_oa_centre(centre)

We also need to set the size of the cropped square in the Fourier domain (called ``radius``)::
    
    radius = 200
    holo.set_oa_radius(radius)
   
    
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

Note that there is generally no need to specify a background (as we would do in inline holography) because the reconstructed field from off-axis holography
already has the background removed.
    
The numpy array ``refocusedField`` is a 2D complex numpy array, to obtain the amplitude and phase as 2D numpy arrays, use::

      amplitude = pyh.amplitude(refocusedField)
      phase = pyh.phase(refocusedField)



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Phase Unwrapping and Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The output from ``Holo.process`` and ``pyholoscope.off_axis_demod`` are complex fields and
``pyholoscope.phase`` returns the wrapped phase. To unwrap the phase use::

    unwrappedPhase = pyh.phase_unwrap(phase)
    
It is sometimes the case that a tilt is added to the phase due to, for example, the coverslip being slightly angled with respect to the microscope slide. 
Assuming that the background image was acquired with the slide removed, using background correction as described above does not help with this. 
Instead use::

    phaseTilt = pyh.obtain_tilt(phase)
    
For this to work, ``phase`` must be unwrapped phase, i.e. the output from  ``pyholoscope.phase_unwrap``. 
This returns a phase correction map to remove the tilt which can then be applied using::

    phaseTilt = pyh.obtain_tilt(phase)

The corrected phase map can then be obtained using::

    correctedPhase = relative_phase(phase, phaseTilt)
     
     
     
             
^^^^^^^^^^^^^^^^
GPU acceleration
^^^^^^^^^^^^^^^^
If a compatible GPU is available, GPU acceleration for off axis demodulation and refocusing using ``Hplo`` is enabled by default. 
Explicitly turn this on or off using::

    holo.set_cuda(True/False)

This requires the CuPy package and a compatible GPU, otherwise processing will revert to the CPU.  

If using the lower level functions, specify ``cuda = True`` when demodulating and refocusing e.g. ::

    reconField = holo.off_axis_demod(hologram, cropCentre, cropRadius, cuda = True)
    holo.refocus(hologram, propagator, cuda = True)


    
    
    