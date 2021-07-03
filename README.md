# PyHoloscope : Holographic Microscopy Utilities for Python
Mainly developed by Mike Hughes, Applied Optics Group, University of Kent. Pull requests welcome.

In process of migrating code from Matlab. Hope to mostly use OpenCV to make this fast. 

First things to implement, for inline holography:
* Generate propagator for angular spectrum method
* Refocus using angular spectrum method
* Cosine window to avoid edge effects
* Focus metrics (Brenner, intensity, maybe others)
* Auto focus (whole image or region) by optimising focus metric

Then implement off-axis holography:
* Complex image recovery by FFT, shifting modulated signal to centre and iFFT
* Auto detect modulation frequency, so don't have to manually choose region to shift

Other things to do in longer timer:
* Port tracking code from Matlab (or maybe try and integrate with TrackPy?)
* Iterative phase recovery for inline holography
* GPU acceleration
