------------------------------------
Off-Axis Holography Using Holo Class
------------------------------------

Off-axis Holography can be performed using the :doc:`holo` of PyHoloscope by setting ``mode = pyholoscope.OFF_AXIS``. This allows
demodulation of the off-axis carrier frequency to recover the quantitative phase as well numerical refocusing. See the 
`Holo Class documentation <holo.html>`_  for a full list of methods and arguments. For code examples see the `Off-axis Holography Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/off_axis_example.py>`_ and the
`Off-axis Holography with Refocusing Example <https://github.com/MikeHughesKent/PyHoloscope/blob/main/examples/off_axis_refocus_example.py>`_ on github.

Alternatively, for more customisability, the low-level functions can be used directly as described in 
`Off-Axis Holography Using Lower-Level Functions <off_axis_low_level.html>`_ .

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using Holo Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
and create an instance of the :doc:`holo`. At a minimum we need to set the mode to off-axis
holography and, if we would like to perform numerical refocusing, provide the physical pixel size and the wavelength, e.g.::

    holo = pyh.Holo(mode = pyh.OFF_AXIS, pixel_size = 2e-6, wavelength = 0.5e-6)
    
We also need to know the spatial frequency of the modulation. We can determine this automatically using::

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

    recon_field = holo.process(hologram)
    
To obtain the amplitude and phase, use::

    amplitude = pyh.amp(recon_field)
    phase = pyh.phase(recon_field) 

If we would like to also refocus to a different depth we can specify this when we create the ``Holo`` object::

    holo = pyh.Holo(mode = pyh.OFF_AXIS, pixel_size = 2e-6, wavelength = 0.5e-6, refocus = True, depth = 0.001)
        
Then when we call::

    recon_field = holo.process(hologram)
    
Both the demodulation and the refocusing will take place in a single step.
    
We can change the refocus depth and whether or not to refocus without recreating the ``Holo`` object using::

    holo.set_depth(depth)
    holo.set_refocus(True)    

Note that the first time a hologram is refocused to a particular depth the process will be slower 
due to the need to create a propagator for that depth. This is particularly noticeable when using
GPU acceleration as the propagator creation will often be the rate limiting step. Subsequent 
refocusing to the same depth will be faster providing no parameters are changed that force 
a new propagator to be created (depth, pixel size, wavelength or grid size). The propagator can
also be pre-computed by calling::

    holo.update_propagator(hologram)
 
in advance.

To correct for a background phase (i.e the phase map of the background hologram), set::

    holo.set_relative_phase(True)
    
or pass ``relative_phase = True`` when creating the ``Holo`` object. You should then call::

    holo.calib_off_axis()
    
to compute the background phase.   

More possibilities for processing and visualising phase are described in :doc:`phase`.
  
    
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Speed Up Using Real FFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

A modest speed-up can often be achieved by setting the ``off_axis_real_fft`` argument to ``True`` when creating the ``Holo`` object, e.g.::

    holo = pyh.Holo(mode = pyh.OFF_AXIS, pixel_size = 2e-6, wavelength = 0.5e-6, off_axis_real_fft = True)

However, this will only work well if the modulated term is close to 45 degrees in the Fourier domain, i.e. the reference
beam is tilted in both the x and y planes (relative to the camera pixel grid) by approximately the same amount. Otherwise,
the modulated term will cross over the zero frequency line and there will be a loss of information.

