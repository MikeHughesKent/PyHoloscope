-----------------
FocusStack Class
-----------------

The ``FocusStack`` class is used to store (but not generate) a stack of images refocused to a range of distances or depths. 
An instance of this class is returned by methods such as ``Holo.depth_stack()``.

The class stores the images internally as a 3D numpy array. The depth corresponding to each index in the array is calculated
based on the ``depth_range`` and ``num_images`` specified at instantiation, such that index 0 is the
smallest depth, index (num_images - 1) is the largest depth, and the intermediate indices correspond to equally spaced
depths in between.

Images can be added either by index (i.e. position in the stack) using ``add_idx`` or by their associated depth,
in which case the ``add_depth`` method calculates the closest index to the specified depth. Both methods will overwrite existing data.
Similarly, images can be retrieved using the ``get_index`` or ``get_depth`` methods, with ``get_depth`` returning the image stored at the 
index corresponding to the depth closest to the requested depth.

.. autoclass:: pyholoscope.FocusStack  
   :members: 

   .. automethod:: __init__
