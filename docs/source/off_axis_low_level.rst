
-----------------------------------------------
Off-Axis Holography Using Lower-Level Functions
-----------------------------------------------

As an alternative to using the :doc:`holo` as described in `Off-Axis Holography Using the Holo Class <off_axis.html>`_ , 
low-level functions can be called directly.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the library::
    
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

    