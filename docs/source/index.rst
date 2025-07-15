.. PyHoloscope documentation master file, created by
   sphinx-quickstart on Thu Nov 10 22:37:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyHoloscope
====================================

PyHoloscope is a Python package for holographic microscopy. It supports inline and off-axis holography, 
including demodulation, numerical refocusing, automatic focusing, and phase correction (e.g. removing tilts). 
It is fast enough for real-time use in imaging GUIs, particularly when the optional GPU and Numba JIT acceleration are used. 

Most of the functionality of PyHoloscope can be accessed via the `Holo class <holo.html>`_. An instance of ``Holo`` is created, processing parameters are set, and then the ``process`` method
is used to process raw holograms. Alternatively, low-level functions can be called directly for more advanced applications where custom processing pipelines are required. 


^^^^^^^^
Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2
   
   installation
   inline
   off_axis
   phase
   autofocus
   sim
   holo
   focus_stack
   propLUT   
   roi
   functions
   contributing

* :ref:`genindex`
