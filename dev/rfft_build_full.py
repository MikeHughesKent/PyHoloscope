# -*- coding: utf-8 -*-
"""
Investigation of building 2D FFT of real image using RFFT

@author: Mike Hughes, Applied Optics Group, University of Kent 
"""

import numpy as np
import time


import context
import pyholoscope as pyh

import matplotlib.pyplot as plt


wavelength = 630e-9
pixelSize = 1e-6
depth = 0.0007
gridSize = 8

prop = pyh.propagator(gridSize, wavelength, pixelSize, depth)
#holo = np.array ( (( 1,3,5,7), (4,6,7,8), (1,2,5.5,6), (5,3,4,5.2)))
holo = np.random.random((2048,2048))
#holo = pyh.load_image(r"..\test\test data\tissue_paper_oa_background.tif")
#holo = np.random.random((8,8))

plt.figure(); plt.title("Raw Hologram")
plt.imshow(holo, cmap='gray')


t1 = time.perf_counter()
holoFT = np.fft.fft2(holo)
print(f"Time for 2D FFT: {round((time.perf_counter() - t1 ) * 1000)} ms")



plt.figure(); plt.title("Log Abs of 2D FFT of Hologram")
plt.imshow(np.log(np.abs(holoFT)), cmap='gray')

plt.figure(); plt.title("Phase 2D FFT of Hologram")
plt.imshow((np.angle(holoFT)), cmap='gray')


t1 = time.perf_counter()
holoRFT = np.fft.rfft2(holo)
print(f"Time for 2D RFFT only: {round((time.perf_counter() - t1 ) * 1000)} ms")
plt.figure(); plt.title("Log Abs of 2D R-FFT of Hologram")
plt.imshow(np.log(np.abs(holoRFT)), cmap='gray')


t1 = time.perf_counter()

holoRFT = np.fft.rfft2(holo)
builtFT = np.zeros_like(holoFT)
builtFT[:, :np.shape(holoRFT)[1]] = holoRFT
builtFT[0, np.shape(holoRFT)[1] - 1 :] = np.conj(np.flipud(holoRFT[0,1:np.shape(holoRFT)[1]]))
builtFT[1:, np.shape(holoRFT)[1] :] = np.conj(np.flip(holoRFT[1:, 1:-1], axis = (0,1)))

print(f"Time for 2D RFFT and Build: {round((time.perf_counter() - t1 ) * 1000)} ms")





plt.figure(); plt.title("Log Abs of 2D Built R-FFT of Hologram")
plt.imshow(np.log(np.abs(builtFT)), cmap='gray')


fft = np.abs(builtFT)
build =  np.abs(holoFT)
half = np.abs(holoRFT)

err = np.mean(np.abs(builtFT) - np.abs(holoFT))
phaseErr = np.mean((np.angle(builtFT) - np.angle(holoFT)))
print(f"Average amplitude error: {err}")
print(f"Average phase error: {phaseErr}")

