---------------
Roi Class
---------------

The ``Roi`` class is used to define a region of interest.


^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: Roi(x, y, width, height)

Creates a region of interest with the specified dimensions.


^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^


.. py:function:: clear_inside(img)

Set pixels in ``img``, a 2D numpy array, to be zero if inside ROI. Returns 2D numpy array.


    
.. py:function:: clear_outside(img)

Set pixels in ``img``, a 2D numpy array, to be zero if outside ROI. Returns 2D numpy array.



.. py:function:: constrain(minX, minY, maxX, maxY)

Adjust the ROI so that it fits within the specified ranges. 


.. py:function:: crop (img)

Extracts a region from an image ``img``, a 2D numpy array.  Returns 2D numpy array.
