---------------
PropLUT Class
---------------

The ``PropLUT`` class used to store a look-up-table (LUT) of angular spectrum propagators.


^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths, [numba = True])


Creates a propagator look up table (LUT) containing angular spectrum propagators
for the specified parameters. ``depthRange`` is a tuple of (min depth, max depth), and ``nDepths``
propagators will be generated equally specifed within this range.



^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^


.. py:function:: propagator(depth) 

Returns the propagator from the LUT which is closest to requested depth. If depth is outside the range of the propagators, returns ``None``.
    
