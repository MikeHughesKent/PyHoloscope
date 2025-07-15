# -*- coding: utf-8 -*-
"""
Tests utilities functions of PyHoloscope.

"""

import unittest 

import numpy as np


from pyholoscope import get8bit, get16bit, amplitude, phase, save_image, save_image16 

from pyholoscope import save_amplitude_image8, save_amplitude_image16, save_phase_image

from pyholoscope import load_image, extract_central, circ_cosine_window, dimensions


class TestUtils(unittest.TestCase):

    def test_gets(self):
        # Test get8it, get16bit, amplitude, phase
        
        img = np.random.randint(0,100, size = (100,100)) + 1j * np.random.randint(0,100, size = (100,100))
        img = img.astype('complex64')
        amp1 = amplitude(img)
        self.assertEqual(amp1.dtype, 'float32')
    
        phase1 = phase(img)
        self.assertEqual(phase1.dtype, 'float32')
    
        amp8, phase8 = get8bit(img)
        self.assertEqual(amp8.dtype, 'uint8')
        self.assertEqual(phase8.dtype, 'uint8')

        amp16, phase16 = get16bit(img)
        self.assertEqual(amp16.dtype, 'uint16')
        self.assertEqual(phase16.dtype, 'uint16')
                
        self.assertTrue(np.allclose(amp16 / 256, amp8, atol = 2))
        self.assertTrue(np.allclose(phase16 / 256, phase8, atol = 2))
        
        
    def test_save_load(self):
        
        img = np.random.randint(0,100, size = (100,100)) + 1j * np.random.randint(0,100, size = (100,100))
        img = img.astype('complex64')
        amp1 = amplitude(img)
        
        save_image(amp1, 'test.tif', autoscale = False)        
        loaded = load_image('test.tif')
        self.assertTrue(np.allclose(amp1, loaded, atol = 1))
        
        save_image(amp1, 'test.tif', autoscale = True) 
        loaded = load_image('test.tif')
        self.assertEqual(np.max(loaded), 255)
        self.assertEqual(np.min(loaded), 0)
        
        save_image16(amp1, 'test.tif', autoscale = False)        
        loaded = load_image('test.tif')
        self.assertTrue(np.allclose(amp1, loaded, atol = 1))
        
        save_image16(amp1, 'test.tif', autoscale = True) 
        loaded = load_image('test.tif')
        self.assertEqual(np.max(loaded), 65535)
        self.assertEqual(np.min(loaded), 0)
        
        save_amplitude_image8(img, 'test.tif') 
        loaded = load_image('test.tif')
        self.assertTrue(np.allclose(get8bit(amplitude(img))[0], loaded, atol = 1))
        
        save_amplitude_image16(img, 'test.tif') 
        loaded = load_image('test.tif')
        self.assertTrue(np.allclose(get16bit(amplitude(img))[0], loaded, atol = 1))

        save_phase_image(img, 'test.tif')
        loaded = load_image('test.tif')
        self.assertTrue(np.allclose(get16bit(img)[1], loaded, atol = 1))


    def test_extract_central(self):
        
        img = np.ones((100,100), dtype = 'complex64')

        boxSize = 50
        cropped = extract_central(img, boxSize)
        self.assertTupleEqual(np.shape(cropped), (boxSize * 2, boxSize * 2))
        
        boxSize = 100
        cropped = extract_central(img, boxSize)
        self.assertTupleEqual(np.shape(cropped), (100,100))
        
        boxSize = 200
        cropped = extract_central(img, boxSize)
        self.assertTupleEqual(np.shape(cropped), (100,100))


    def test_circ_cosine_window(self):
        
        imgSize = 100
        circleRadius = 40
        skinThickness = 10
        window = circ_cosine_window(imgSize, circleRadius, skinThickness)
        self.assertEqual(window[0,0], 0)
        self.assertEqual(window[50,50], 1)
        self.assertTupleEqual(np.shape(window), (imgSize, imgSize))        
        

    def test_dimensions(self):

        w = 2
        h = 4
        a = np.zeros((h, w))
        assert dimensions(np.zeros((h, w))) == (w, h)
        assert dimensions((w,h)) == (w, h)
        assert dimensions((w)) == (w,w) 
        

if __name__ == '__main__':
    import context
    unittest.main()