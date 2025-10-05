---------------
PropLUT Class
---------------

The ``PropLUT`` class is used to create and store a look-up-table (LUT) of propagators
for subsequent faster refocusing to multiple depths using the angular spectrum method. An 
instance of this class is returned by methods such as ``Holo.make_propagator_LUT()``.

The propagators are created at instantiation, and then are extracted as required using the
``propagator()`` method.



.. autoclass:: pyholoscope.PropLUT 
   :members: 

   .. automethod:: __init__

