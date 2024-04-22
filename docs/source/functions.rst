----------------------------------
Function Reference
----------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^
Classes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: Holo()

Provides object-oriented access to core functionality of PyHoloscope. See `Holo class <holo.html>`_ for details.

.. py:function:: FocusStack(img, depthRange, nDepths)

Stores a refocus stack, allowing convenient acccess to each image and ability to write stack to a file. See `FocusStack class <focus_stack.html>`_ for details.

.. py:function:: PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths)

Stores a propagator look up table for faster refocusing across multple depths. See `PropLUT class <propLUT.html>`_ for details.

.. py:function:: Roi(x, y, width, height)

Region of interest, a rectangle with top left co-ordinates (``x``, ``y``) with width ``width`` and height ``height``. ``crop`` method is used to extract the ROI
from an image and ``constrain`` method is used to limit co-ordinates to adjust the ROI to fit within an image. See `Roi class <roi.html>`_ for details.



^^^^^^^^^^^^^^^^^^^^^^^^^
General Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.fourier_plane_display

.. autofunction:: pyholoscope.pre_process

^^^^^^^^^^^^^^^^^^^
Off-axis Holography
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.off_axis_demod

.. autofunction:: pyholoscope.off_axis_find_mod

.. autofunction:: pyholoscope.off_axis_find_crop_radius

.. autofunction:: pyholoscope.off_axis_predict_mod

.. autofunction:: pyholoscope.off_axis_predict_tilt_angle


^^^^^^^^^^^^^^^^^^^^
Numerical Refocusing
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.coarse_focus_search

.. autofunction:: pyholoscope.focus_score

.. autofunction:: pyholoscope.find_focus

.. autofunction:: pyholoscope.focus_score_curve

.. autofunction:: pyholoscope.propagator

.. autofunction:: pyholoscope.propagator_numba

.. autofunction:: pyholoscope.refocus

.. autofunction:: pyholoscope.refocus_and_score

.. autofunction:: pyholoscope.refocus_stack


^^^^^^^^^^^^^^^^^^^^^^^^^
Phase Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.mean_phase

.. autofunction:: pyholoscope.obtain_tilt

.. autofunction:: pyholoscope.phase_gradient

.. autofunction:: pyholoscope.phase_gradient_amp

.. autofunction:: pyholoscope.phase_gradient_dir

.. autofunction:: pyholoscope.phase_unwrap

.. autofunction:: pyholoscope.relative_phase

.. autofunction:: pyholoscope.relative_phase_self
 
.. autofunction:: pyholoscope.synthetic_DIC


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utility Functions : Loading and Saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.load_image

.. autofunction:: pyholoscope.save_image

.. autofunction:: pyholoscope.save_amplitude_image8

.. autofunction:: pyholoscope.save_amplitude_image16

.. autofunction:: pyholoscope.save_phase_image


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utility Functions : Image Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.get8bit

.. autofunction:: pyholoscope.get16bit

.. autofunction:: pyholoscope.magnitude

.. autofunction:: pyholoscope.amplitude

.. autofunction:: pyholoscope.phase


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utility Functions : Windowing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.extract_central

.. autofunction:: pyholoscope.circ_window

.. autofunction:: pyholoscope.circ_cosine_window

.. autofunction:: pyholoscope.square_cosine_window


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utility Functions : Other
^^^^^^^^^^^^^^^^^^^^^^^^^s^^^^^^^^^^^^

.. autofunction:: pyholoscope.invert

.. autofunction:: pyholoscope.dimensions


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyholoscope.sim.off_axis






