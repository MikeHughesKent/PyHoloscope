[![Documentation Status](https://app.readthedocs.org/projects/pyholoscope/badge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![PyHoloscope Logo](/res/pyholoscope_logo_.png)
# PyHololoscope: Fast Holographic Microscopy for Python

PyHoloscope is a Python package for holographic microscopy, providing perfomant reconstruction of inline and off-axis holograms.

PyHoloscope is designed to be:
* Fast (for Python) - optmised for CPU using Numpy, Scipy and Numba, with GPU support via CuPy
* Easy to Use - a simple object-oriented API gives high performance without low-level tweaks
* Suitable for Live Imaging - can be used as the back-end of holographic microscopy GUIs as well as for offline processing

Full documentation is on [Read the docs](https://pyholoscope.readthedocs.io/en/latest/index.html). 

Also see the examples in the [examples folder](https://github.com/MikeHughesKent/PyHoloscope/tree/main/examples).

Contributions to the package (new features, tests or documentation) as very welcome, please see the roadmap below and post in the discussion if you are working on something, or [get in touch](mailto:m.r.hughes@kent.ac.uk).

Development is co-ordinated by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes/)' lab in the 
[Applied Optics Group](https://research.kent.ac.uk/applied-optics), Physics & Astronomy, University of Kent. 

If you are interested in academic applications we are happy to help, post in the dicussion. If you would like help using PyHoloscope for commercial purposes, consultancy is available, please contact [Mike Hughes](mailto:m.r.hughes@kent.ac.uk) in the first instance.

[Join the mailing list](https://groups.google.com/g/pyholoscope) to hear about releases, updates and bug fixes.

## Installation

```bash
pip install pyholoscope
```
For more details see the [Installation Guide](https://pyholoscope.readthedocs.io/en/latest/installation.html).

## Features

### General
* Object-oriented interface
* Choice of single or double precision
* Support for CUDA compatible GPUs
* Optional use of Numba JIT compiler
* Supports non-square holograms

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

## Roadmap of Planned Developments (help welcome!)

- [] Phase recovery for inline holography
- [] Improved optimisation for speed/multiple back-ends
- [] Support phase-shifting holography
- [] Support coded aperture/multi-depth phase recovery
- [] Support forward scattering and inference 
- [] Support particle tracking 
- [] Deep learning for focusing
- [] Targeted support for edge computing (e.g. Raspberry Pi)

## Requirements
* Numpy
* Scipy
* PIL
* OpenCV
* Scikit-Image
* Matplotlib
* Numba (optional, for JIT acceleration)
* CuPy (optional, for GPU)
