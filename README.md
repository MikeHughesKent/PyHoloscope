# PyHoloscope : Holographic Microscopy Utilities for Python
*Mainly developed by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes/), Applied Optics Group, University of Kent. Pull requests welcome.*

A on-going project to develop tools for holographic microscopy in Python. The aim is to make this optimised, fast and suitable for real-time use, including GPU acceleration. There is already a python library called [HoloPy](https://github.com/manoharan-lab/holopy) which implements a lot of useful functionality, so check that out if you need something that works now.

As of now, this is very in-development, there are lots of bugs and not much documentation. If you would like to use the library, there are some examples in the test folder that will get you started.

The following is currently implemented for general holography:
* Refocus using angular spectrum method 
* Cosine window to avoid edge effects - (done for circular window for bundle holograpy, need rectangular for general purpose)
* Generate stack of images at different focal positions
* Focus metrics (Brenner, Sobel, Peak Intensity, DarkFocus, SobelVariance 
* Auto focus whole image or ROI by optimising focus metric, through fast bounded search and (optionally) initial coarse search to narrow search range.
* Generate LUT of propagators for faster auto-focus or repeated geberation of focus stacks.

For off-axis holography:
* Complex image recovery by FFT, shifting modulated signal to centre and iFFT
* Auto detect modulation frequency
* Remove background phase 
* Phase relative to some region of interest 
* Synthetic DIC/phase contrast from QPI (partially done, some bugs)

Other things to do in longer timer:
* Port tracking code from Matlab (or maybe try and integrate with TrackPy?)
* Iterative phase recovery for inline holography
* GPU acceleration
* Phase recovery techniques (from inline holography)
