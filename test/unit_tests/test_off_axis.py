# -*- coding: utf-8 -*-
"""
Tests off-axis holography functions of PyHoloscope

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import math
import unittest

import numpy as np
import matplotlib.pyplot as plt

import context

import pyholoscope as pyh
import pyholoscope.sim
from pyholoscope.utils import circ_cosine_window


class TestOffAxis(unittest.TestCase):

    
    def test_predict_tilt_angle(self):
        """ Generates simulated hologram then determine tilt angle 
        from FFT and checks
        """
        
        gridSize2 = 256   
        gridSize1 = 512  
        
        angle = math.radians(3.6)

        pixel_size = 2e-6
        wavelength = 500e-9
        
        object_field = np.ones((gridSize2, gridSize1))
        test_hologram = pyh.sim.off_axis(object_field, wavelength, pixel_size, angle, rotation =  math.pi/4)
       
        measured_angle = pyh.off_axis_predict_tilt_angle(test_hologram, wavelength, pixel_size)
        
        self.assertAlmostEqual(angle, measured_angle, places = 2)



    def test_find_mod(self):
          """Generates hologram and compares detected peak in FFT with predicted
          peak position.
          """
          
          gridSize2 = 512   
          gridSize1 = 1024 
        
          angle = math.radians(3)
          rotation = .22 * math.pi 
          pixel_size = 2e-6
          wavelength = 500e-9
         
          object_field = np.ones((gridSize2, gridSize1))
          test_hologram = pyh.sim.off_axis(object_field, wavelength, pixel_size, angle, rotation = rotation)
         
          measured_peak_loc = pyh.off_axis_find_mod(test_hologram)        
          predicted_peak_loc = pyh.off_axis_predict_mod(wavelength, pixel_size, (gridSize1, gridSize2), angle, rotation = rotation)
            
          self.assertAlmostEqual(predicted_peak_loc[0], measured_peak_loc[0], delta = 1)
          self.assertAlmostEqual(predicted_peak_loc[1], measured_peak_loc[1], delta = 1)
          
          measured_peak_distance = math.sqrt(measured_peak_loc[0]**2 + measured_peak_loc[1]**2)
          predicted_peak_dist = pyh.off_axis_predict_mod_distance(wavelength, pixel_size, (gridSize1, gridSize2), angle, rotation = rotation)
          
          self.assertAlmostEqual(measured_peak_distance, predicted_peak_dist, delta = 1)
          
          # Check it works when modulation peak in 2nd quadrant of FFT
          rotation = .7 * math.pi 
          test_hologram = pyh.sim.off_axis(object_field, wavelength, pixel_size, angle, rotation = rotation)
           
          measured_peak_loc = pyh.off_axis_find_mod(test_hologram)        
          predicted_peak_loc = pyh.off_axis_predict_mod(wavelength, pixel_size, (gridSize1, gridSize2), angle, rotation = rotation)
          
          self.assertAlmostEqual(predicted_peak_loc[0], measured_peak_loc[0], delta = 1)
          self.assertAlmostEqual(predicted_peak_loc[1], measured_peak_loc[1], delta = 1)
          
          measured_peak_distance = math.sqrt(measured_peak_loc[0]**2 + measured_peak_loc[1]**2)
          predicted_peak_dist = pyh.off_axis_predict_mod_distance(wavelength, pixel_size, (gridSize1, gridSize2), angle, rotation = rotation)
            
          self.assertAlmostEqual(measured_peak_distance, predicted_peak_dist, delta = 1)
      
        
    def test_off_axis_demod(self):
          """ Check standard demo with and without window
          """
        
          gridSize2 = 512   
          gridSize1 = 512
          pixel_size = 1e-6
          wavelength = 550e-9
          rotation = math.pi / 4
          angle = math.radians(15)
                 
          x = 100
          y = 70
          w = 120
          h = 150
        
          object_field = np.zeros((gridSize2, gridSize1))
          object_field[y:y +h, x:x+w] = 1
          
          test_hologram = pyh.sim.off_axis(object_field, wavelength, pixel_size, angle, rotation = rotation)
         
          cropCentre = pyh.off_axis_find_mod(test_hologram)
          cropRadius = pyh.off_axis_find_crop_radius(test_hologram)
           
          # Check recon has a bright square, same as object          
          scaleFactor = cropRadius[0] / gridSize2 * 2 
          x2 = round(x * scaleFactor)
          y2 = round(y * scaleFactor)
          w2 = round(w * scaleFactor)
          h2 = round(h * scaleFactor)

          # no window
          recon = pyh.off_axis_demod(test_hologram, cropCentre, cropRadius)
         
          # compare mean value in the square to mean value somewhere elese (that should be zero)
          assert np.mean(pyh.amplitude(recon[y2:y2+h2, x2:x2+w2])) > 100 * np.mean(pyh.amplitude(recon[4*y2:4*y2+h2, x2:x2+w2]))

          # with window
          window = circ_cosine_window(cropRadius[0] * 2, cropRadius[0] - 10, 10)
          recon = pyh.off_axis_demod(test_hologram, cropCentre, cropRadius, mask = window)
            
          # compare mean value in the square to mean value somewhere elese (that should be zero)
          assert np.mean(pyh.amplitude(recon[y2:y2+h2, x2:x2+w2])) > 100 * np.mean(pyh.amplitude(recon[4*y2:4*y2+h2, x2:x2+w2]))
       
    def test_off_axis_demod_full(self):
          """ Checks demod and return of image same size as hologram
          """
        
          gridSize2 = 512   
          gridSize1 = 512 
          pixel_size = 1e-6
          wavelength = 550e-9
          rotation = 3 * math.pi / 4
          angle = math.radians(9)
                 
          x = 100
          y = 70
          w = 120
          h = 150
        
          object_field = np.zeros((gridSize2, gridSize1))
          object_field[y:y +h, x:x+w] = 1
          
          test_hologram = pyh.sim.off_axis(object_field, wavelength, pixel_size, angle, rotation = rotation)
         
          cropCentre = pyh.off_axis_find_mod(test_hologram)
          cropRadius = pyh.off_axis_find_crop_radius(test_hologram)
           
          # no window
          recon = pyh.off_axis_demod(test_hologram, cropCentre, cropRadius, returnFull = True)
         
          # compare mean value in the square to mean value somewhere elese (that should be zero)
          assert np.mean(pyh.amplitude(recon[y:y+h, x:x+w])) > 100 * np.mean(pyh.amplitude(recon[4*y:4*y+h, x:x+w]))

          # with window
          window = circ_cosine_window(cropRadius[0] * 2, cropRadius[0] - 10, 10)
          recon = pyh.off_axis_demod(test_hologram, cropCentre, cropRadius, mask = window, returnFull = True)
            
          # compare mean value in the square to mean value somewhere elese (that should be zero)
          assert np.mean(pyh.amplitude(recon[y:y+h, x:x+w])) > 100 * np.mean(pyh.amplitude(recon[4*y:4*y+h, x:x+w]))


    def test_off_axis_demod_rectangular(self):
          """ Checks demod and return of image for non-square hologram
          """
        
          gridSize2 = 512   
          gridSize1 = 1024
          pixel_size = 1e-6
          wavelength = 550e-9
          rotation = math.pi / 4
          angle = math.radians(15)
                 
          x = 100
          y = 70
          w = 120
          h = 150
        
          object_field = np.zeros((gridSize2, gridSize1))
          object_field[y:y +h, x:x+w] = 1
          
          test_hologram = pyh.sim.off_axis(object_field, wavelength, pixel_size, angle, rotation = rotation)
         
          cropCentre = pyh.off_axis_find_mod(test_hologram)
          cropRadius = pyh.off_axis_find_crop_radius(test_hologram)
         
          # no window
          recon = pyh.off_axis_demod(test_hologram, cropCentre, cropRadius, returnFull = True)
         
          # compare mean value in the square to mean value somewhere elese (that should be zero)
          assert np.mean(pyh.amplitude(recon[y:y+h, x:x+w])) > 100 * np.mean(pyh.amplitude(recon[4*y:4*y+h, x:x+w]))

          
                

if __name__ == '__main__':
    unittest.main()