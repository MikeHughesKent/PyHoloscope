# PyHoloscope : Holographic Microscopy Utilities for Python
Mainly developed by Mike Hughes, Applied Optics Group, University of Kent. Pull requests welcome.

A project to develop tools for holographic microscopy in Python. The aim is to make this optimised, fast and suitable for real-time use, including GPU acceleration. There is already a python library called [HoloPy](https://github.com/manoharan-lab/holopy) which implements a lot of useful functionality, so check that out if you need something that works now.

As of now, this is very in-development, there are lots of bugs and not much documentation.

Have begun implementing:
* Generate propagator for angular spectrum method (done)
* Refocus using angular spectrum method (done)
* Cosine window to avoid edge effects - (done for circular window for bundle holograpy, need rectangular for general purpose)
* Focus metrics (Brenner, Sobel, Peak Intensity done)
* Auto focus whole image or ROI by optimising focus metric done (done).
* Generate LUT of propagators for faster auto-focus (done).

All needs documenting properly.

Then have implemented some functions for off-axis holography:
* Complex image recovery by FFT, shifting modulated signal to centre and iFFT (done)
* Auto detect modulation frequency, so don't have to manually choose region to shift (done)
* Remove background phase (done)
* Phase relative to some region of interest (in progress)
* Synthetic DIC/phase contrast from QPI (partially done, some bugs)

Other things to do in longer timer:
* Port tracking code from Matlab (or maybe try and integrate with TrackPy?)
* Iterative phase recovery for inline holography
* GPU acceleration
* Phase recovery techniques (from inline holography)
