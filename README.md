# PyHoloscope : Holographic Microscopy Utilities for Python

PyHoloscope is a Python package for holographic microscopy. 

Development is by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes/)' lab in the 
[Applied Optics Group](https://research.kent.ac.uk/applied-optics), School of Physics & Astronomy, University of Kent. Bug reports, contributions and pull requests are welcome. 

The package was originally developed mostly for applications in Mike Hughes' lab, including compact and fibre-based holography, 
but it is general enough for a variety of applications. It supports inline and off-axis holography, including numerical refocusing and phase correction (e.g. removing tilts). The package is designed to be fast enough for use in imaging GUIs as well as for offline research. GPU acceleration is supported. 

Full documentation is on [Read the docs](https://pyholoscope.readthedocs.io/en/latest/index.html). Also see the examples in the examples folder and the test folder.

Development is active and there may be bugs and future changes to the API.

## Features

### Off Axis Holography
* Complex image recovery by FFT, shifting modulated signal to centre and iFFT (GPU acceleration available)
* Auto detect modulation frequency
* Predict tilt angle from modulation frequency

### General and Inline holography
* Refocus using angular spectrum method 
* Cosine window to avoid edge effects - (done for circular window for bundle holograpy, need rectangular for general purpose)
* Generate stack of images at different focal positions
* Focus metrics (Brenner, Sobel, Peak Intensity, DarkFocus, SobelVariance)
* Auto focus whole image or ROI by optimising focus metric, through fast bounded search and (optionally) initial coarse search to narrow search range.
* Generate LUT of propagators for faster auto-focus or repeated generation of focus stacks.

### Phase Visualation
* Remove background phase 
* Remove 1D phase tilt
* Phase relative to some region of interest 
* Synthetic DIC/phase contrast

## Planned Developments (help welcome!)
* Support non-square holograms
* Better auto-focusing
* Improved optimisation for speed
* Port tracking code from Matlab (or maybe try and integrate with TrackPy?)
* Phase recovery for inline holography

