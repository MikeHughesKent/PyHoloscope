----------------------------------
Automatic Depth Determination
----------------------------------

The optimal refocus depth for a hologram can be determined using trial refocuses, assessed using a focus metric. 
The package supports an exhaustive search or a faster golden section based optimisation method. The focus can be
optimised for the entire image or for a region of interest (ROI). Faster determination for a ROI can be achieved
by only refocusing a subset of the image surrounding the ROI. It is also possible to create a look up table of propagators
to provide a further speed-up.

^^^^^^^^^^^^^^^^^
Getting Started 
^^^^^^^^^^^^^^^^^

Begin by importing the package::

    import pyholoscope as pyh
    
We will assume we have a hologram ``hologram`` stored as a 2D numpy array. This can either be a raw inline hologram, in 
which case ``hologram`` will be real, or a demodulated off-axis hologram, in which case ``hologram`` will be complex.
    
For refocusing we need to define the image size, in pixels, pixel size, and wavelength, for example::

    imageSize = 512
    pixelSize = 2e-6
    wavelength = 0.5e-6
    
We also need to define the range of depths to search over as a tuple of ``(minDepth, maxDepth)``, and select
the focus metric to use, for this example we will choose the 'Sobel' metric::
    
    depthRange = (0, 0.001)
    method = 'Sobel'
    
We then call ``find_focus``::
    
    focusDepth = pyh.find_focus(hologram, wavelength, pixelSize, depthRange, method)
   
This returns the best refocus depth.

If we would like to find the best focus for a specific region of the image only, we define a ROI such as::

    x = 20
    y = 30
    w = 40
    h = 40
    focusRoi = pyh.Roi(x,y,w,h)
    
and pass this as an optional argument::

    focusDepth = pyh.find_focus(hologram, wavelength, pixelSize, depthRange, method, roi = focusRoi )
    
The entire hologram will still be refocused, but only the focus metric will only be applied to the ROI, so there is no speed-up from doing this.

A speed-up can be obtained by refocusing only the ROI plus a small margin around it. This is activated by specifying the margin in pixels::

    focusDepth = pyh.find_focus(hologram, wavelength, pixelSize, depthRange, method, roi = focusRoi, margin = 20)

If the hologram has not had a background subtracted or a window previously applied, we can also request these are applied prior to refocusing by passing them as optional arguments::

    window = pyh.circ_cosine_window(imageSize, imageSize / 2, 20)
    focusDepth = pyh.find_focus(hologram, wavelength, pixelSize, depthRange, method, roi = focusRoi, margin = 20, background = backgroundImg, window = window)      


^^^^^^^^^^^^^
Focus Metrics
^^^^^^^^^^^^^

The available options for the focus metric are:

- Peak - Takes the largest pixel value
- Var - Takes the standard deviation
- Brenner - Applies the Brenner edge detection filter and takes the mean       
- Sobel - Applies a Sobel edge detection filter and takes the mean
- SobelVariance - Applies a Sobel edge detection filter and takes the standard deviation
- DarkFocus - Takes the standard deviation of the quadrature sum of the vertical and horizontal image gradients

The result from each of these metric is negated so that the best focus depth has the smallest score.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using a Propagator Look Up Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When determining the focal depth, the rate limiting step can sometimes be the generation of angular spectrum method propagators for each trial refocus
depth. When repeated autofocusing will be performed over a similar depth range, and with the same size ROI (or whole image), a speed-up can be achieved
by pre-generating the required propagators.

To do this, create an instance of the PropLUT class. We need to specify the range of refocus depths and the number of depths within this range that propagators
will be generated for. When searching for focus, we can search within a smaller range than this, but not larger. Increasing the number of depths within
the range will increase the precision of the autofocusing.::

    imgSize = 256
    wavelength = 0.5e-6
    pixelSize = 2e-6
    depthRange = (0, 0.1)
    nDepths = 100
    lut = pyh.PropLUT(imgSize, wavelength, pixelSize, depthRange, nDepths)
    
The lut is then passed to the ``find_focus`` function::

    focusDepth = pyh.find_focus(hologram, wavelength, pixelSize, depthRange, method, propagatorLUT = lut)      

If also attempting to speed-up by refocusing only a ROI (i.e. the margin is specified) then it is necessary to create a propagator LUT of the correct size for 
this ROI + margin. This must currently be done manually by adjusing the ``gridSize`` input to be the ROI.
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Performing a prior coarse search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard method of finding focus uses a golden-section based minimisation algorithm which can be prone to finding local minima. To reduce this problem, 
an initial exhaustive coarse search can be made to identify the most likely depth region containing focus, and the standard method is then applied only within 
this region. We specify the number of regularly spaced points within the depth range to check the focus for, a fine search using golden section is then performed
in an interval around the point with the best focus score, for example::

    focusDepth = pyh.find_focus(hologram, wavelength, pixelSize, depthRange, method, coarseSearchInterval = 10)      

