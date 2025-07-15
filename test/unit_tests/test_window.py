# -*- coding: utf-8 -*-
"""
Tests windowing functionality of PyHoloscope

"""

import unittest 

import numpy as np

import context                    # Relative paths

import pyholoscope as pyh



class TestWindow(unittest.TestCase):


    # Experimental Parameters
    wavelength = 630e-9
    pixelSize = 1e-6
    depth = 0.0127
    
    # Create image
    hologram = np.zeros((200,200))

    def test_create(self):

        # Check that autoWindow results in a window being created when we call update_auto_window
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,                
                        depth = self.depth,
                        autoWindow = True)
        
        holo.update_auto_window(self.hologram)
        self.assertTupleEqual(np.shape(holo.window), np.shape(self.hologram))


        # Check that default window is the same as one created manually 
        windowLowLevel = pyh.square_cosine_window(self.hologram, (np.shape(self.hologram)[1] / 2, np.shape(self.hologram)[0] / 2), 10)
        assert np.array_equal(windowLowLevel, holo.window) == True


    def test_autowindow(self):
        
        # Check that autoWindow creates a window of the specified radius and defualt thickness
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,                
                        depth = self.depth,
                        autoWindow = True,
                        windowRadius = 100)
        holo.update_auto_window(self.hologram)
        
        windowLowLevel = pyh.square_cosine_window(self.hologram, 100, 10)
        assert np.array_equal(windowLowLevel, holo.window) == True


        # Check that autoWindow creates a window of the specified radius and thickness
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,                
                        depth = self.depth,
                        autoWindow = True,
                        windowRadius = 100,
                        windowThickness = 20)
        holo.update_auto_window(self.hologram)
        
        windowLowLevel = pyh.square_cosine_window(self.hologram, 100, 20)
        assert np.array_equal(windowLowLevel, holo.window) == True


    def test_resize(self):
        # Check that the wrong window size gets resized with a warning displayed
        
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,                
                        depth = self.depth)
        
        windowLowLevel = pyh.square_cosine_window(int(np.shape(self.hologram)[0] / 2), 100, 20)
        holo.set_window(windowLowLevel)
        
        recon = holo.process(self.hologram)
        windowLowLevel = pyh.square_cosine_window(int(np.shape(self.hologram)[0] / 2), 100, 20)
        holo.set_window(windowLowLevel)
        recon = holo.process(self.hologram)
        
        self.assertNotEqual(recon.any(), None)


    def test_no_autowindow(self):
        # Check that if autoWindow not set, update_auto_window does not create a window
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,                
                        depth = self.depth)
        
        holo.update_auto_window(self.hologram)
        assert holo.window is None


    def test_autowindow_setter(self):
        # Check we can set autoWindow using setter
        
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,                
                        depth = self.depth)
        
        holo.set_auto_window(True)
        holo.update_auto_window(self.hologram)
        assert np.shape(holo.window) == np.shape(self.hologram)
        
    def test_autowindow_process(self):
        
        # Check that autoWindow results in a window being created when we call process
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        autoWindow = True)
        
        recon = holo.process(self.hologram)
        assert np.shape(holo.window) == np.shape(self.hologram)
        
    def test_downsampling(self):


        # Check that autoWindow results in the correct size window being created when 
        # we call process with downsampling
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        downsample = 2,
                        autoWindow = True)
        
        recon = holo.process(self.hologram)
        self.assertTupleEqual(np.shape(holo.window), np.shape(recon))
        
        # Check that autoWindow results in the correct size window being created when 
        # we call update_auto_window with downsampling
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        downsample = 2,
                        autoWindow = True)
        
        holo.update_auto_window(self.hologram)
        wind = holo.window
        recon = holo.process(self.hologram)
        
        self.assertTupleEqual(np.shape(wind), np.shape(recon))

    def test_manual(self):
        # Check that we can set a manually created window
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth)

        windowLowLevel = pyh.square_cosine_window(self.hologram, 100, 10)
        
        holo.set_window(windowLowLevel)
        recon1 = holo.process(self.hologram)
        
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        autoWindow = True,
                        windowRadius = 100)
        recon2 = holo.process(self.hologram)            
        
        assert np.array_equal(recon1, recon2) == True
        
        
    def test_create_window(self):

        # Check that we can a window using create_window
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth)
        
        holo.create_window(self.hologram, 100, 10)
        
        recon1 = holo.process(self.hologram)
        
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        autoWindow = True,
                        windowRadius = 100)
        recon2 = holo.process(self.hologram)
        
        assert np.array_equal(recon1, recon2) == True

    def test_post_refocus(self):

        # Check that we can can apply a post refocusing window
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        autoWindow = True,
                        postWindow = True)
        
        recon1 = holo.process(self.hologram)
        
        holo = pyh.Holo(mode = pyh.INLINE_MODE, 
                        wavelength = self.wavelength, 
                        pixelSize = self.pixelSize,
                        depth = self.depth,
                        autoWindow = True)
        recon2 = holo.process(self.hologram) 
        
        recon2.imag = recon2.imag * holo.window
        recon2.real = recon2.real * holo.window
        recon2[recon2 == -0+0j] = 0j
        
        assert np.array_equal(recon1, recon2) == True

if __name__ == '__main__':
    unittest.main()