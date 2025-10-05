==================================
Phase Processing and Visualisation
==================================

The raw output from numerical refocusing or off-axis demodulation is a complex
numpy array representing the complex field. The amplitude and phase of the field
are the amplitude and angle of the complex numbers in the array, respectively, 
and can be extracted using :func:`pyholoscope.amp` and :func:`pyholoscope.phase`.

PyHoloscope has several functions for further processing of the phase. A common
requirement is to subtract a reference phase, for example acquired with nothing 
in the field of view. This can be done automatically when using the :class:`Holo`
class by providing a background hologram, using the ``set_background`` method and then
``set_relative_phase(True)`` (or by setting the ``background`` and ``relative_phase``
parameters at instantiation).

Alternatively, this can be done manually using the :func:`pyholoscope.relative_phase`
function. For example, assume we have off-axis demodulated a hologram, to obtain ``hologram_demod``,
for example by using the :class:`Holo` class. We then also demodulate the
background image to obtain ``background_demod`` in the same way. Then::

    correctedImage = pyholoscope.relative_phase(hologram_demod, background_demod)

returns a complex array with the reference phase from the background hologram
subtracted. It is also possible to provide the phase maps (i.e. ``pyholoscope.phase(hologram_demod)``
and ``pyholoscope.phase(background_demod)``) instead, in which case the corrected phase will
be returned. The function assumes that the inputs are phase maps if the datatype is not complex.

^^^^^^^^^^^^^^^^
Phase Unwrapping
^^^^^^^^^^^^^^^^

An unwrapped phase map can be created by passing the wrapped phase map to
:func:`pyholoscope.phase_unwrap`. Note that this function accepts either a 
phase map or the complex field, but the output unwrapped phase map is always
real, i.e. the unwrapped phase cannot be represented as a complex field. The phase
unwrapping uses the method shipped with skimage, based on:

Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path”, Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002

^^^^^^^^^^^^
Tilt Removal
^^^^^^^^^^^^

The way in which a sample is mounted may lead to further distortions of the phase
map that are not accounted for by subtraction of a reference phase. This often
manifests as a tilt. This can be removed by first estimating the tilt from the 
unwrapped phase and then subtracting it, using :func:`pyholoscope.obtain_tilt` and
the removing this using :func:`pyholoscope.relative_phase`. Note that
:func:`pyholoscope.obtain_tilt` must be passed the unwrapped phase map, e.g.
from :func:`pyholoscope.phase_unwrap`, otherwise the tilt will not be 
found correctly.

For example, assuming we have a complex field in ``hologram_demod``::

    unwrapped_phase = pyholoscope.phase_unwrap(hologram_demod)
    tilt = pyholoscope.obtain_tilt(unwrapped_phase)
    corrected_phase = pyholoscope.relative_phase(unwrapped_phase, tilt)
    
This returns the unwrapped and tilt-corrected phase map in ``corrected_phase``.   
It is possible to use :func:`pyholoscope.relative_phase` twice, once to remove
the background phase distortion due to the system (in which case the tilt should
be estimated from this background-corrected phase) and then again to correct 
the tilt.

^^^^^^^^^^^^^^^^^^^
Phase Stabilisation
^^^^^^^^^^^^^^^^^^^

In live imaging or video recording, the global phase will tend to drift over time. To avoid this
visual effect, the phase can be referenced to a part of the image which is known not to contain any object using
:func:`pyholoscope.relative_phase_self`. Define a region of interest (ROI) using :class:`Roi` and then pass this
along with the complex field or phase map, for example::

    roi = pyholoscope.Roi(x=20, y=20, w = 100, h = 100)
    corrected_phase = pyholoscope.relative_phase_self(hologram_demod, roi)
    
Alternatively, do not pass a ROI to make the phase relative to the mean phase across the entire
image. This gives good results only when the object makes up a small fraction of the image.
   
The function works with both the complex field and phase maps, but will not be effective when
the complex field contains wrapped phase or uncorrected tilt or other distortions. It is normally
best to correct these issues first using the relative phase, tilt removal and phase unwrapping functions, 
and then pass the unwrapped, corrected phase to :func:`pyholoscope.relative_phase_self`.
 

^^^^^^^^^^^^^^^^^^^
Visualisation
^^^^^^^^^^^^^^^^^^^

PyHoloscope includes two functions, :func:`pyholoscope.phase_gradient` and :func:`pyholoscope.synthetic_DIC` to
generate a phase gradient image and an approximation to a DIC image, respectively.

Both functions accept either the complex field or phase map (i.e. a complex or
float array) and work with wrapped phase. Artefacts due to wrapping are avoided
by correcting large jumps in phase. :func:`pyholoscope.phase_gradient_amp` is used internally
by :func:`pyholoscope.phase_gradient`, and does not remove the wrapping 
(on unwrapped phase maps either functions can be used and return identical values).

A third function, :func:`pyholoscope.phase_gradient_dir` can be used to return a directional
gradient map. This map is returned as a complex array, with the amplitudes being
the amplitude of the gradient at each point, and the angles being the direction. This
function should be provided with the unwrapped phase.

^^^^^^^^^^^^^^^^^^^
Saving Phase Maps
^^^^^^^^^^^^^^^^^^^

Wrapped phase maps can be saved to image files using :func:`pyholoscope.save_phase_image`, which 
saves the phase map as a 16 bit tif, in which a pixel value of 0 corresponds to a phase of 0
and a pixel values of 65535 corresponds to 2pi. The function accepts either phase maps or complex fields.
If a phase map is provided, any phase values outside of the [0, 2pi] interval will be wrapped to this
interval.

An unwrapped phase map, stored as a float array, can be saved using 
:func:`pyholoscope.save_image` or :func:`pyholoscope.save_image16`, to save as 8 or 16 bit images
respectively. Note that the images will be autoscaled so that the minimum phase value is 0 and the max
is either 255 or 65535, respectively. To avoid this, pass ``autoscale = False``, in which case the phase
values will be cast directly to the relevant data type, in which case it is the caller's 
responsibility to ensure the range is sensible. 
