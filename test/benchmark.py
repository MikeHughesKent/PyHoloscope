# -*- coding: utf-8 -*-
"""
PyHoloscope Speed Tests

"""

from matplotlib import pyplot as plt
import numpy as np
import time
import context

import timeit

import pyholoscope as pyh

# Propagator parameters
grid_sizes = [256, 512, 1024, 2048]
wavelength = 0.5e-9
pixel_size = 0.5e-6
depth = 0.001

# Look up table
depth_range = (0.5 * depth, 2 * depth)
num_depths = 10

print("------------------")
print("Timings (ms):")
print("------------------")

print("-------------------------------")
print("Propagator Generation No Numba:")
print("-------------------------------")
for grid_size in grid_sizes:
    testcode = "pyh.propagator(grid_size, wavelength, pixel_size, depth, precision = 'single', use_numba = False)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("----------------------------------")
print("Propagator Generation Using Numba:")
print("----------------------------------")
for grid_size in grid_sizes:
    pyh.propagator_numba((grid_size, grid_size), wavelength, pixel_size, depth)
    testcode = "pyh.propagator((grid_size, grid_size), wavelength, pixel_size, depth)"    
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("--------------------------------")
print("Propagator Table Build No Numba:")
print("--------------------------------")
for grid_size in grid_sizes:
    testcode = "pyh.PropLUT(grid_size, wavelength, pixel_size, depth_range, num_depths, use_numba = False)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("----------------------------------")
print("Propagator Table Build With Numba:")
print("----------------------------------")
for grid_size in grid_sizes:
    testcode = "pyh.PropLUT(grid_size, wavelength, pixel_size, depth_range, num_depths, use_numba = True)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("----------------------------")
print("Refocus by Angular Spectrum:")
print("----------------------------")
for grid_size in grid_sizes:
    prop = pyh.propagator((grid_size, grid_size), wavelength, pixel_size, depth)
    img = np.random.random((grid_size, grid_size))
    testcode = "pyh.refocus(img, prop)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("------------------------------------")
print("Complex Refocus by Angular Spectrum:")
print("------------------------------------")
for grid_size in grid_sizes:
    prop = pyh.propagator((grid_size, grid_size), wavelength, pixel_size, depth)
    img = np.random.random((grid_size, grid_size)) + 1j * np.random.random(
        (grid_size, grid_size)
    )
    testcode = "pyh.refocus(img, prop)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("----------------------")
print("Off-Axis Demodulation:")
print("----------------------")
for grid_size in grid_sizes:
    img = np.random.random((grid_size, grid_size))
    testcode = "pyh.off_axis_demod(img, (grid_size / 4, grid_size / 4), (grid_size / 8, grid_size / 8))"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("----------------------------------")
print("Numerical refocusing (Holo Class):")
print("----------------------------------")
for grid_size in grid_sizes:
    holo = pyh.Holo(
        mode=pyh.INLINE, wavelength=wavelength, pixel_size=pixel_size, depth=depth
    )
    img = np.random.random((grid_size, grid_size))
    out = holo.process(img)
    testcode = "holo.process(img)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("--------------------------------------------------")
print("Numerical refocusing with Background (Holo Class):")
print("--------------------------------------------------")
for grid_size in grid_sizes:
    back = np.random.random((grid_size, grid_size))
    holo = pyh.Holo(
        mode=pyh.INLINE,
        background=back,
        wavelength=wavelength,
        pixel_size=pixel_size,
        depth=depth,
    )
    img = np.random.random((grid_size, grid_size))
    out = holo.process(img)
    testcode = "holo.process(img)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("-----------------------------------------------------")
print("Numerical refocusing with Normalisation (Holo Class):")
print("-----------------------------------------------------")
for grid_size in grid_sizes:
    back = np.random.random((grid_size, grid_size))
    holo = pyh.Holo(
        mode=pyh.INLINE,
        normalise=back,
        wavelength=wavelength,
        pixel_size=pixel_size,
        depth=depth,
    )
    img = np.random.random((grid_size, grid_size))
    out = holo.process(img)
    testcode = "holo.process(img)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")


print("----------------------------------------------")
print("Numerical refocusing with Window (Holo Class):")
print("----------------------------------------------")
for grid_size in grid_sizes:
    back = np.random.random((grid_size, grid_size))
    holo = pyh.Holo(
        mode=pyh.INLINE,
        autoWindow=True,
        wavelength=wavelength,
        pixel_size=pixel_size,
        depth=depth,
    )
    img = np.random.random((grid_size, grid_size))
    out = holo.process(img)
    testcode = "holo.process(img)"
    t = timeit.timeit(stmt=testcode, number=10, globals=globals())
    print(f"Size {grid_size} x {grid_size} : {round(t * 100, 2)}")
