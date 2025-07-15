# -*- coding: utf-8 -*-
"""
PyHoloscope Speed Tests

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import context

import timeit

import pyholoscope as pyh

# Propagator parameters
gridSizes = [256, 512, 1024, 2048]
wavelength = 0.5e-9
pixelSize = 0.5e-6
depth = 0.001

# Look up table
depthRange = (0.5 * depth, 2 * depth)
nDepths = 10

print("------------------")
print("Timings (ms):")
print("------------------")

print("-------------------------------")
print("Propagator Generation No Numba:")
print("-------------------------------")
for gridSize in gridSizes:      
    testcode = "pyh.propagator(gridSize, wavelength, pixelSize, depth, precision = 'single')"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")


print("----------------------------------")
print("Propagator Generation Using Numba:")
print("----------------------------------")
for gridSize in gridSizes: 
    pyh.propagator_numba((gridSize, gridSize), wavelength, pixelSize, depth)
    testcode = "pyh.propagator_numba((gridSize, gridSize), wavelength, pixelSize, depth)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")


print("--------------------------------")
print("Propagator Table Build No Numba:")
print("--------------------------------")
for gridSize in gridSizes: 
    testcode = "pyh.PropLUT(gridSize, wavelength, pixelSize, depthRange, nDepths)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")
    

print("----------------------------------")
print("Propagator Table Build With Numba:")
print("----------------------------------")
for gridSize in gridSizes: 
    testcode = "pyh.PropLUT(gridSize, wavelength, pixelSize, depthRange, nDepths, numba = True)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")    


print("----------------------------")
print("Refocus by Angular Spectrum:")
print("----------------------------")
for gridSize in gridSizes: 
    prop = pyh.propagator_numba((gridSize, gridSize), wavelength, pixelSize, depth)
    img = np.random.random((gridSize, gridSize))
    testcode = "pyh.refocus(img, prop)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")  
    

print("------------------------------------")
print("Complex Refocus by Angular Spectrum:")
print("------------------------------------")
for gridSize in gridSizes: 
    prop = pyh.propagator_numba((gridSize, gridSize), wavelength, pixelSize, depth)
    img = np.random.random((gridSize, gridSize)) + 1j * np.random.random((gridSize, gridSize))
    testcode = "pyh.refocus(img, prop)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")   


print("----------------------")
print("Off-Axis Demodulation:")
print("----------------------")
for gridSize in gridSizes: 
    img = np.random.random((gridSize, gridSize)) 
    testcode = "pyh.off_axis_demod(img, (gridSize / 4, gridSize / 4), (gridSize / 8, gridSize / 8))"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")       


print("----------------------------------")
print("Numerical refocusing (Holo Class):")
print("----------------------------------")
for gridSize in gridSizes: 
    holo = pyh.Holo(mode = pyh.INLINE, wavelength = wavelength, pixelSize = pixelSize, depth = depth)
    img = np.random.random((gridSize, gridSize)) 
    out = holo.process(img)    
    testcode = "holo.process(img)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")    


print("--------------------------------------------------")
print("Numerical refocusing with Background (Holo Class):")
print("--------------------------------------------------")
for gridSize in gridSizes: 
    back = np.random.random((gridSize, gridSize)) 
    holo = pyh.Holo(mode = pyh.INLINE, background = back, wavelength = wavelength, pixelSize = pixelSize, depth = depth)
    img = np.random.random((gridSize, gridSize)) 
    out = holo.process(img)    
    testcode = "holo.process(img)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")    



print("-----------------------------------------------------")
print("Numerical refocusing with Normalisation (Holo Class):")
print("-----------------------------------------------------")
for gridSize in gridSizes: 
    back = np.random.random((gridSize, gridSize)) 
    holo = pyh.Holo(mode = pyh.INLINE, normalise = back, wavelength = wavelength, pixelSize = pixelSize, depth = depth)
    img = np.random.random((gridSize, gridSize)) 
    out = holo.process(img)    
    testcode = "holo.process(img)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")    
    

print("----------------------------------------------")
print("Numerical refocusing with Window (Holo Class):")
print("----------------------------------------------")
for gridSize in gridSizes: 
    back = np.random.random((gridSize, gridSize)) 
    holo = pyh.Holo(mode = pyh.INLINE, autoWindow = True, wavelength = wavelength, pixelSize = pixelSize, depth = depth)
    img = np.random.random((gridSize, gridSize)) 
    out = holo.process(img)    
    testcode = "holo.process(img)"
    t =  timeit.timeit(stmt=testcode,number=10,globals=globals())
    print(f"Size {gridSize} x {gridSize} : {round(t * 100,2)}")        