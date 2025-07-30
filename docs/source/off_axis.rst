----------------------------------
Off-Axis Holography Basics
----------------------------------
Off-axis Holography is performed using the :doc:`holo` of PyHoloscope by setting ``mode = pyholoscope.OFFAXIS``. This allows
demodulation of the off-axis carrier frequency to recover the quantitative phase as well numerical refocusing. See the 
`Holo Class documentation <holo.html>`_  for a full list of methods and arguments. For code examples see the `Off-axis Holography Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/off_axis_example.py>`_ and the
`Off-axis Holography with Refocusing Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/off_axis_refocus_example.py>`_ on github.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using Holo Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
and create an instance of the :doc:`holo`. At a minimum we need to set the mode to off-axis
holography and, if we would like to perform numerical refocusing, provide the physical pixel size and the wavelength, e.g.::

    holo = pyh.Holo(mode = pyh.OFF_AXIS, pixel_size = 2e-6, wavelength = 0.5e-6)
    
We also need to know the spatial frequency of the modulation. We can determine this automtically using::

    holo.calib_off_axis(background_img)         
    
where we have provided a background hologram to use for this purpose. It is possible 
to use the image hologram as well for this purpose, but this may be unreliable if there
is another strong spatial frequency.  If a background image has first been set using::

    holo.set_background(background_img)
    
or by passing ``background = background_img`` when creating the ``Holo`` object, then
this will be used for the calibration if ``calib_off_axis`` is called with no argument.

Alternatively the demodulation parameters can be specified manually using::

    holo.set_off_axis_mod(crop_centre, crop_radius)
   
where ``crop_centre`` is a tuple of (x,y), giving the pixel location of the centre of the modulation peak in the FFT of the hologram, 
and ``crop_radius`` is half the size of the box around the modulation centre which is demodulated.  ``crop_radius`` can also
be a tuple giving the x and y radii of an ellipse, for cases where the hologram is non-square.  

We can then demodulate to obtain the complex field using::

    recon_field = holo.process()
    
To obtain the amplitude and phase, use::

    amplitude = pyh.amp(recon_field)
    phase = pyh.phase(recon_field) 

If we would like to also refocus to a different depth we can specify this when we create the ``Holo`` object::

    holo = pyh.Holo(mode = pyh.OFFAXIS, pixel_size = 2e-6, wavelength = 0.5e-6, refocus = True, depth = 0.001)
        
Then when we call::

    recon_field = holo.process()
    
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

More possibilities for processing and visualising phase are described in :doc:`phase`.
  
    

    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Speed Up Using Real FFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

A modest speed-up can often be achieved by setting the ``off_axis_real_fft`` argument to ``True`` when creating the ``Holo`` object, e.g.::

    holo = pyh.Holo(mode = pyh.OFFAXIS, pixel_size = 2e-6, wavelength = 0.5e-6, off_axis_real_fft = True)

However, this will only work well if the modulated term is close to 45 degrees in the Fourier domain, i.e. the reference
beam is tilted in both the x and y planes (relative to the camera pixel grid) by approximately the same amount. Otherwise,
the modulated term will cross over the zero frequency line and there will be a loss of information.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative to using the ``Holo`` class, low-level functions can be called directly. Begin by importing the library::
    
    import pyholoscope as pyh
    
We will assume we have a hologram ``hologram`` and a background image ``background_img`` which are both square 2D numpy arrays of the same size. 
If we do not know the modulation frequency in advance we can use::

    crop_centre = pyh.off_axis_find_mod(background_img)
    crop_radius = pyh.off_axis_find_crop_radius(background_img)  
    
We can then demodulate using::

    recon_field = pyh.off_axis_demod(hologram, crop_centre, crop_radius)
    
To remove the background, recover the background field using::

    background_field = pyh.off_axis_demod(background, crop_centre, crop_radius)  
    
Remove the background phase (for example due to aberrations in the imaging system) using::

    corrected_field = pyh.relative_phase(recon_field, background_field)
    
The numpy array ``corrected_field`` is complex, to obtain the amplitude and phase, use::

    amplitude = pyh.amp(recon_field)
    phase = pyh.phase(recon_field) 
  
If we would like to numerically refocus, we first define a propagator for use with the angular spectrum method. 
This requires specification of the hologram size, wavelength, pixel size and the depth we wish to refocus to::

    grid_size = crop_radius * 2
    wavelength = 0.5e-6
    pixel_size = 2e-6
    depth = 1e-3
    prop = pyh.propagator(grid_size, wavelength, pixel_size, depth)
    
Note here that the ``grid_size`` is the size of the reconstructed field following demodulation which is smaller than the original image. 
The pixel size must also be specified as the pixel size in the reconstructed field, not the pixel size in the original hologram. 
Pixel size, wavelength and depth must be in the same units.
 
We can then refocus using::

    refocused_img = pyh.refocus(corrected_field, propagator)

The numpy array ``refocused_field`` is a 2D complex numpy array, to obtain the amplitude and phase as 2D numpy arrays, use::

      amplitude = pyh.amp(refocused_field)
      phase = pyh.phase(refocused_field)

As when using the ``Holo`` class, for cases when you know the reference beam is tilted in the both the x and y planes, 
it is possible to improve speed by using only a real FFT by passing ``real_fft = True`` to the ``off_axis_demod`` function, e.g.::

    recon_field = pyh.off_axis_demod(hologram, crop_centre, crop_radius, real_fft=True)

``real_fft = True`` must also be passed to the ``off_axis_find_mod`` and ``off_axis_find_crop_radius`` functions to 
correctly find the peak location in the real FFT.

    