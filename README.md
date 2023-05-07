# PyHoloscope : Holographic Microscopy for Python

PyHoloscope is a Python package for holographic microscopy image processing, both inline and off-axis. It is under development but close to the first release.

PyHoloscope is designed to be:
* Fast - optmised both for CPU and GPU, using Cupy and Numba
* Easy to Use - an object-oriented interface allows you to set up the processing scheme and then
process raw holograms with a single method call

Full documentation is on [Read the docs](https://pyholoscope.readthedocs.io/en/latest/index.html). Also see the examples in the examples folder and the test folder.

Development is led by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes/)' lab in the 
[Applied Optics Group](https://research.kent.ac.uk/applied-optics), School of Physics & Astronomy, University of Kent. 
Help testing and developing the package is welcome, please get in touch.


## Features

### Off Axis Holography
* Complex image recovery by FFT, shifting modulated signal to centre and iFFT (GPU acceleration available)
* Auto detect modulation frequency
* Predict tilt angle from modulation frequency

### General and Inline holography
* Refocus using angular spectrum method 
* Cosine window to avoid edge effects 
* Generate stack of images at different focal positions
* Apply focus metrics (Brenner, Sobel, Peak Intensity, DarkFocus, SobelVariance)
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

