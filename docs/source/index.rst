.. PyHoloscope documentation master file, created by
   sphinx-quickstart on Thu Nov 10 22:37:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyHoloscope
====================================

PyHoloscope is a Python package for holographic microscopy. 

Development is by `Mike Hughes <https://research.kent.ac.uk/applied-optics/hughes/>`_'s lab in the 
`Applied Optics Group <https://research.kent.ac.uk/applied-optics>`_, School of Physics and Astronomy, University of Kent. 
Bug reports, contributions and pull requests are welcome. 

The package was originally developed mostly for applications in Mike Hughes lab, including compact and fibre-based holography, 
but it is general enough for a variety of applications. 
It supports inline and off-axis holography, including numerical refocusing and phase correction (e.g. removing tilts). 

The package is designed to be fast enough for use in imaging GUIs as well as for offline research. GPU acceleration is supported. 
Development is active and there may be bugs and future changes to the API.

.. toctree::
   :maxdepth: 2
   
   inline
   off_axis
   holo
   autofocus
   propLUT
   roi
   functions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
