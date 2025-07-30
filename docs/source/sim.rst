----------------------------------
Simulation
----------------------------------
PyHoloscope includes a function for simulating off-axis holograms. The simulation
module must be imported explicitly::

    import pyholoscope.sim 
    
To simulate the hologram, we provide a complex 2D numpy array to represent the
complex field from the object, as well as the wavelength and camera pixel size.
We also specifiy the angle of the reference beam with respect to the object
beam. Optionally, we can also specify the orientation of this angle with 
respect to the x-axis in radians, by default this is pi/4 (45 degrees) which is
commonly used in off-axis holography to maximise the field of view by placing
the modulation along the diagonal.
    
For this example, we generate a uniform object::

    objectField = np.ones((256, 256))  
  
and then generate a hologram for a tilt angle of 0.087 rad (5 degrees), assuming 3 micron
pixels and 500 nm wavelength::  
  
    hologram = pyholoscope.sim.off_axis(objectField, 
                                        wavelength = 500e-9, 
                                        pixel_size = 3e-6, 
                                        tilt_angle = 0.087)
                             
We can then check the off-axis modulation using::

    predictedTilt = pyh.off_axis_predict_tilt(hologram, 
                                              wavelength = 500e-9,
                                              pixel_size = 3e-6)

which returns approximately the tilt angle specified when creating the
simulated hologram. We can also reconstruct the object field using the
off axis functionality of PyHoloscope.

A further optional argument allows the ``OPD`` to be specified, this is an
optical path difference between the object and reference beams. This will 
make no observable change to a single hologram (other than a slight shift of
the modulation) but it is useful when simulating multiple holograms at 
different wavelengths to capture coherence effects (such as in OCT).                                              
                             