# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:31:49 2022

@author: AOG
"""

import numpy as np
import timeit
import time
import cupy as cp

im = np.random.rand(1024,1024)

t1 = time.perf_counter()
im2 = np.fft.fft2(im)

print(time.perf_counter() - t1)

im =cp.random.rand(1024,1024)      
t1 = time.perf_counter()
im2 = cp.fft.fft2(im)
print(time.perf_counter() - t1)
