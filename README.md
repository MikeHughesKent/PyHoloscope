# PyHoloscope : Holographic Microscopy Utilities for Python
Mainly developed by Mike Hughes, Applied Optics Group, University of Kent. Pull requests welcome.

In process of migrating code from Matlab. Hope to mostly use OpenCV to make this fast. 

Have begun implemening:
* Generate propagator for angular spectrum method - done
* Refocus using angular spectrum method - done
* Cosine window to avoid edge effects - done for circular window for bundle holograpy, need rectangular for general purpose
* Focus metrics (Brenner, Sobel, Peak Intensity done)
* Auto focus whole image by optimising focus metric done, need to implement ROI

All needs documenting properly.

Then implement off-axis holography:
* Complex image recovery by FFT, shifting modulated signal to centre and iFFT
* Auto detect modulation frequency, so don't have to manually choose region to shift

Other things to do in longer timer:
* Port tracking code from Matlab (or maybe try and integrate with TrackPy?)
* Iterative phase recovery for inline holography
* GPU acceleration
