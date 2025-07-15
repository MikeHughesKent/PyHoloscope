# -*- coding: utf-8 -*-
"""
Tests speed increase from using Numba and qaudrant version of angular spectrum
propagator generator in PyHoloscope.

"""

from matplotlib import pyplot as plt
import numpy as np
import time
import context

import pyholoscope as pyh

# Propagator parameters
gridSize = 1024
wavelength = 0.5e-9
pixelSize = 0.5e-6
depth = 0.001

# Look up table
depthRange = (0.5 * depth, 2 * depth)
nDepths = 10

print("------------------")
print("Timings (ms):")
print("------------------")


t1 = time.perf_counter()
prop2 = pyh.propagator(gridSize, wavelength, pixelSize, depth, precision = 'single')
print("No Numba: ", round((time.perf_counter() - t1) * 1000,1))


# Since we now run prop at import, the two Numba runs should be roughly the same
t1 = time.perf_counter()
prop3 = pyh.propagator_numba((gridSize, gridSize), wavelength, pixelSize, depth)
print("Numba Run 1: ", round((time.perf_counter() - t1) * 1000,1))


t1 = time.perf_counter()
prop4 = pyh.propagator_numba((gridSize, gridSize), wavelength, pixelSize, depth)
print("Numba Run 2: ", round((time.perf_counter() - t1) * 1000,1))


t1 = time.perf_counter()
lut = pyh.PropLUT(gridSize, wavelength, pixelSize, depthRange, nDepths)
print("Table Build Without Numba: ", round((time.perf_counter() - t1) * 1000,1))


t1 = time.perf_counter()
lut = pyh.PropLUT(gridSize, wavelength, pixelSize, depthRange, nDepths, numba = True)
print("Table Build With Numba: ", round((time.perf_counter() - t1) * 1000,1))


t1 = time.perf_counter()
prop5 = lut.propagator(depth)
print("Table Lookup: ", round((time.perf_counter() - t1) * 1000,2))

# Check all methods give the same propagator
print("------------------")
print("Numerical Errors:")
print("------------------")
print("Numba Method Error: ", np.mean(np.angle(prop4 - prop2)))
print("Lookup Error: ", np.mean(np.angle(prop5 - prop2)))



