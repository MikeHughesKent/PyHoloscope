---------------
PropLUT Class
---------------

The ``PropLUT`` class is used to create and store a look-up-table (LUT) of propagators
for subsequent faster refocusing to multiple depths using the angular spectrum method. An 
instance of this class is returned by methods such as ``Holo.make_propagator_LUT()``.

The propagators are created at instantiation, and then are extracted as required using the
``propagator()`` method.


^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths, [numba = True])


Creates a propagator look up table (LUT) containing angular spectrum propagators for holograms
of size ``imgSize``, either an int for square images or tuple of ints for width and height for rectangular images.
``wavelength`` and ``pixelSize`` are the physical parameters for the hologram, 
and are specified in the same units as ``depthRange``, a tuple of (min depth, max depth). A total of ``nDepths``
propagators will be generated equally specifed within this range (inlcuding the min and max values). 
The Numba JIT will be used for speed-up by default unless ``numba = False``.



^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^


.. autofunction::  pyholoscope.prop_lut.PropLUT.propagator
 