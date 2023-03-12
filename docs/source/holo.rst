----------
Holo Class
----------

The Holo class is the preferred way to access the core functionality of PyHoloscope. Examples are provided

^^^^^^^
Methods
^^^^^^^

.. py:function:: Holo(mode, wavelength, pixelSize, **kwargs)

Intantiation of a Holo object. ``mode`` determines the pipeline of processing to apply to images, either ``PyHoloscope.offaxis`` or ``PyHoloscope.inline``.
``wavelength`` is the light wavelength, ``pixelSize`` is the camera pixel size (as projected onto the object plan if magnification is present). 
``wavelength`` and ``pixelSize`` should be in the same units. Return an instance of Holo.

.. py:function:: set_depth(depth)





    