
![PyHoloscope Logo](/res/pyholoscope_logo_.png)
# PyHololoscope: Holographic Microscopy for Python

PyHoloscope is a Python package for holographic microscopy image processing, both inline and off-axis. It is under development but reasonably stable and close to the first release.

PyHoloscope is designed to be:
* Fast (for Python) - optmised for CPU using Numpy, Scipy and Numba, with GPU support via CuPy
* Easy to Use - a simple object-oriented API gives high performance without low-level tweaks
* For Live Imaging - can be used as the back-end of holographic microscopy GUIs as well as for offline processing

Full documentation is on [Read the docs](https://pyholoscope.readthedocs.io/en/latest/index.html). 

Also see the examples in the [examples folder](https://github.com/MikeHughesKent/PyHoloscope/tree/main/examples) as well as the tests in the [test folder](https://github.com/MikeHughesKent/PyHoloscope/tree/main/test).

Development is mainly by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes/)' lab in the 
[Applied Optics Group](https://research.kent.ac.uk/applied-optics), School of Physics & Astronomy, University of Kent. I'm happy to collaborate with academic users to help your use case, and if you would like help using PyHoloscope for commercial purposes, 
consultancy is available, please contact [Mike Hughes](mailto:m.r.hughes@kent.ac.uk). 

Help testing and developing the package is welcome, please [get in touch](mailto:m.r.hughes@kent.ac.uk).

[Join the mailing list](https://groups.google.com/g/pyholoscope) to hear about releases, updates and bug fixes.



## Features

### General
* Object-oriented interface
* Choice of single or double precision
* Support for CUDA compatible GPUs
* Optional use of Numba JIT compiler

### Off Axis Holography
* Quantitatave phase and amplitude recovery from off-axis hologram
* Auto detect off-axis modulation frequency
* Predict tilt angle from modulation frequency

### Numerical Refocusing (Inline and Off-axis Holgoraphy)
* Refocus holograms or complex fields using the angular spectrum method 
* Choice of cosine windows to reduce edge effects 
* Generate stack of images at different refocus depths
* Apply focus metrics (Brenner, Sobel, Peak Intensity, DarkFocus, SobelVariance)
* Auto focus whole image or ROI by optimising focus metric, through fast bounded search and (optionally) initial coarse search to narrow search range.
* Generate LUT of propagators for faster auto-focus or repeated generation of focus stacks.

### Phase Visualation
* Remove background phase 
* Remove phase tilt
* Show phase relative to region of interest 
* Generate phase contrast image
* Generate synthetic DIC image

## Planned Developments (help welcome!)

### Short-term (Before release 1.0.0)
* Support holograms with odd side lengths
* Better auto-focusing
* FFTW integration for faster CPU-only refocusing

### Long-term
* Improved optimisation for speed
* Support phase-shifting holography
* Support coded aperture/multi-depth phase recovery
* Support forward scattering and inference (or intergrate with HoloPy)
* Port tracking code from Matlab (or integrate with TrackPy)
* Phase recovery for inline holography
* Deep learning for focusing
* Targeted support for edge computing (e.g. Raspberry Pi)

## Requirements
* Numpy
* Scipy
* PIL
* OpenCV
* Scikit-Image
* Matplotlib
* Numba (optional, for JIT acceleration)
* CuPy (optional, for GPU)
