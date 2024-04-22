-----------------
FocusStack Class
-----------------

The ``FocusStack`` class is used to store (but not generate) a stack of images refocused to a range of distances or depths. 
An instance of this class is returned by methods such as ``Holo.depth_stack()``.

The class stores the images internally as a 3D numpy array. The depth corresponding to each index in the array is calculated
based on the ``depthRange`` and ``numImages`` specified at instantiation, such that index 0 is the
smallest depth, index (numImages - 1) is the largest depth, and the intermediate indices correspond to equally spaced
depths inbetweeen.

Images can be added either by index (i.e. position in the stack) using ``add_idx`` or by their associated depth,
in which case the ``add_depth`` method calculates the closest index to the specified depth. Both methods will overwrite existing data.
Similarly, images can be retrieved using the ``get_index`` or ``get_depth`` methods, with ``get_depth`` returning the image stored at the 
index corresponding to the depth closest to the requested depth.


^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: FocusStack(img, depthRange, numImages)

``img`` is a numpy array of the same size and data type as the images to be stored (the pixel values are irrelevant, this
is used purely to initialise a 3D array to store the images). ``depthRange`` is a tuple of (min_depth, max_depth)
that the class will store, and ``numImages`` is the number of images that will be stored, equally spaced within the
depth range.

Note that FocusStack does not create the refocused images, it is used to store the output of a function such as ``Holo.depth_stack()``.


^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.FocusStack.add_idx
     
.. autofunction:: pyholoscope.FocusStack.add_depth

.. autofunction:: pyholoscope.FocusStack.depth_to_index

.. autofunction:: pyholoscope.FocusStack.get_index
  
.. autofunction:: pyholoscope.FocusStack.get_depth
 
.. autofunction:: pyholoscope.FocusStack.get_depth_intensity

.. autofunction:: pyholoscope.FocusStack.get_index_intensity

.. autofunction:: pyholoscope.FocusStack.index_to_depth

.. autofunction:: pyholoscope.FocusStack.write_intensity_to_tif
     
.. autofunction:: pyholoscope.FocusStack.write_phase_to_tif

