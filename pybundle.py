# -*- coding: utf-8 -*-
"""
PyBundle is an open source Python package for image processing of
fibre bundle images.

Classes:
    PyBundle : Bundle image processing
    Mosaic   : High speed mosaicking

@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""


import numpy as np
import math
import time
from matplotlib import pyplot as plt

import cv2 as cv
from skimage.transform import hough_circle, hough_circle_peaks


class PyBundle:
    
   
    
    def __init__(self):
        pass
       

    # Applies Gaussian filter to image
    def gFilter(img, filterSize):
        
        kernelSize = round(filterSize * 6)           # Kernal size needs to be larger than sigma
        kernelSize = kernelSize + 1 - kernelSize%2   # Kernel size must be odd
        imgFilt = cv.GaussianBlur(img,(kernelSize,kernelSize), filterSize)
        return imgFilt
    
        
    # Locates bundle in an image by thresholding and searching for largest
    # connected region. Returns tuple of (centreX, centreY, radius)
    def findBundle(img, **kwargs):
        
        filterSize = kwargs.get('filterSize', 4)
        
        # Filter to minimise effects of structure in bundle
        kernelSize = round(filterSize * 6)           # Kernal size needs to be larger than sigma
        kernelSize = kernelSize + 1 - kernelSize%2   # Kernel size must be odd
        imgFilt = cv.GaussianBlur(img,(kernelSize,kernelSize), filterSize)
        
        # Threshold to binary and then look for connected regions
        thres, imgBinary = cv.threshold(imgFilt,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
        num_labels, labels, stats, centroid  = cv.connectedComponentsWithStats(imgBinary, 8, cv.CV_32S)
        
        # Region 0 is background, so find largest of other regions
        sizes = stats[1:,4]
        biggestRegion = sizes.argmax() + 1
        
        # Find distance from centre to each edge and take minimum as safe value for radius
        centreX = round(centroid[biggestRegion,0]) 
        centreY = round(centroid[biggestRegion,1])
        radius1 = centroid[biggestRegion,0] - stats[biggestRegion,0]
        radius2 = centroid[biggestRegion,1] - stats[biggestRegion,1]
        radius3 = -(centroid[biggestRegion,0] - stats[biggestRegion,2]) + stats[biggestRegion,0]
        radius4 = -(centroid[biggestRegion,1] - stats[biggestRegion,3]) + stats[biggestRegion,1]
        radius = round(min(radius1, radius2, radius3, radius4))          
              
        return centreX, centreY, radius
    
        
    # Extracts a square around the bundle using specified co-ordinates  
    def cropRect(img,loc):
        cx = loc[0]
        cy = loc[1]
        rad = loc[2]
        imgCrop = img[cy-rad:cy+ rad, cx-rad:cx+rad]
        
        # Correct the co-ordinates of the bundles so that they
        # are correct for new cropped image
        newLoc = [rad,rad,loc[2] ]
   
        return imgCrop, newLoc
    
        
    # Finds the mask and applies it to set all values outside
    def maskAuto(img, loc):
        imgMasked = np.multiply(img, PyBundle.getMask(img,loc))
        return imgMasked
    
    # Sets all pixels outside bundle to 0
    def mask(img, mask):
        imgMasked = np.multiply(img, mask)
        return imgMasked
    
    # Returns location of bundle (tuple of centreX, centreY, radius) and a
    # a mask image with all pixel inside of bundle = 1, and those outside = 2
    def locateBundle(img):
        loc = PyBundle.findBundle(img)
        imgD, croppedLoc = PyBundle.cropRect(img,loc)
        mask = PyBundle.getMask(imgD, croppedLoc)
        return loc, mask
    
    # Sequentially crops image to bundle, applies Gaussian filter and then
    # sets pixels outside bundle to 0
    def cropFilterMask(img, loc, mask, filterSize, **kwargs):
        
        resize = kwargs.get('resize', -1)
        
        img = PyBundle.mask(img, mask)
        img = PyBundle.gFilter(img, filterSize)
        img, newLoc = PyBundle.cropRect(img, loc)
        if resize > 0:
            img = cv.resize(img, (resize,resize))
        
        
        return img


    # Returns a mask, 1 inside bundle, 0 outside bundle
    def getMask(img, loc):
        cx = loc[0]
        cy = loc[1]
        rad = loc[2]
        mY,mX = np.meshgrid(range(img.shape[0]),range(img.shape[1]))
        
        m = np.square(mX - cx) +  np.square(mY - cy)   
        imgMask = np.transpose(m < rad**2)
         
        return imgMask
    
    
    # Find cores in bundle image using Hough transform. This generally
    # does not work as well as findCores and is a lot slower!
    def findCoresHough(img, **kwargs):
       
        scaleFac = kwargs.get('scaleFactor', 2)
        cannyLow = kwargs.get('cannyLow', .05)
        cannyHigh = kwargs.get('cannyHigh', .8)
        estRad = kwargs.get('estRad', 1)
        minRad = kwargs.get('minRad', np.floor(max(1,estRad)).astype('int'))
        maxRad = kwargs.get('maxRad', np.floor(minRad + 2).astype('int'))
        minSep = kwargs.get('minSep', estRad * 2)
        darkRemove = kwargs.get('darkRemove', 2)
        gFilterSize = kwargs.get('filterSize', estRad / 2)

        
        imgR = cv.resize(img, [scaleFac * np.size(img,0),  scaleFac * np.size(img,1)] ).astype(float)
              
        # Pre filter with Gaussian and Canny
        imgF = PyBundle.gFilter(imgR, gFilterSize*scaleFac) 
        imgF = imgF.astype('uint8')
        edges = cv.Canny(imgF,cannyLow,cannyHigh)
        
        # Using Scikit-Image Hough implementation, trouble getting CV to work
        radii = range(math.floor(minRad * scaleFac),math.ceil(maxRad * scaleFac))
        circs = hough_circle(edges, radii, normalize=True, full_output=False)
 
 
        minSepScaled = np.round(minSep * scaleFac).astype('int')
       
        for i in range(np.size(circs,0)):
            circs[i,:,:] = np.multiply(circs[i,:,:], imgF) 
        
        
        accums, cx, cy, radii = hough_circle_peaks(circs, radii, min_xdistance = minSepScaled, min_ydistance = minSepScaled)

        # Remove any finds that lie on dark points
        meanVal = np.mean(imgF[cy,cx])
        stdVal = np.std(imgF[cy,cx])
        removeCore = np.zeros_like(cx)
        for i in range(np.size(cx)):
            if imgF[cy[i],cx[i]] < meanVal - darkRemove * stdVal :
                removeCore[i] = 1
        cx = cx[removeCore !=1]        
        cy = cy[removeCore !=1]      

        cx = cx / scaleFac    
        cy = cy / scaleFac     
            
        
        return cx,cy, imgF, edges, circs
    
      
    

    # Find cores in bundle image using regional maxima. Generally fast and 
    #accurate
    def findCores(img, coreSpacing):
       
        
        # Pre-filtering helps to minimse noise and reduce efffect of
        # multimodal patterns
        imgF = PyBundle.gFilter(img, coreSpacing/5)
       
        # Find regional maximum by taking difference between dilated and original
        # image. Because of the way dilation works, the local maxima are not changed
        # and so these will have a value of 0
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(coreSpacing,coreSpacing))
        imgD = cv.dilate(imgF, kernel)
        imgMax = 255 - (imgF - imgD)  # we need to invert the image

        # Just keep the maxima
        thres, imgBinary = cv.threshold(imgMax,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Dilation step helps deal with mode patterns which have led to multiple
        # maxima within a core, the two maxima will end up merged into one connected
        # region
        elSize = math.ceil(coreSpacing / 3)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elSize,elSize))
        imgDil = cv.dilate(imgBinary, kernel)

        # Core centres are centroids of connected regions
        nReg, p1, p2, centroid = cv.connectedComponentsWithStats(imgDil, 8, cv.CV_32S)
        cx = centroid[1:,0]  # The 1st entry is the background
        cy = centroid[1:,1]
                
        return cx, cy
    
    
        
        
    # Extract a central square from an image
    def extractCentral(img, boxSize):
        w = np.shape(img)[0]
        h = np.shape(img)[1]

        cx = w/2
        cy = h/2
        boxSemiSize = min(cx,cy,boxSize)
        
        imgOut = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
        return imgOut
    
      
    
    
    
    
        
##############################################################################        
class Mosaic:
    
    CROP = 0
    EXPAND = 1
    SCROLL = 2
    
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3  
    
      
    def __init__(self, mosaicSize, **kwargs):
        
        self.mosaicSize = mosaicSize
        self.prevImg = []

        # If the default value of -1 is used for the following
        # sensible values will be selected after the first image
        # is received
        self.resize = kwargs.get('resize', -1)
        self.templateSize = kwargs.get('templateSize', -1)
        self.refSize = kwargs.get('refSize', -1)
        self.cropSize = kwargs.get('cropSize', -1)
        
        self.blend = kwargs.get('blend', True)
        self.blendDist = kwargs.get('blendDist', 40)
        self.minDistForAdd = kwargs.get('mindDistForAdd', 5)
        self.currentX = kwargs.get('initialX', round(mosaicSize /  2))
        self.currentY = kwargs.get('initialY', round(mosaicSize /  2))
        self.expandStep = kwargs.get('expandStep', 50)
        self.imageType = kwargs.get('imageType', -1)

        
        # These are created the first time they are needed
        self.mosaic = []
        self.mask = []
        self.blendMask = []
       
        # Initial values
        self.lastShift = [0,0]
        self.lastXAdded = 0
        self.lastYAdded = 0
        self.nImages = 0        
        self.imSize = -1  # -1 tell us to read this from the first image
        
      
        self.boundaryMethod = kwargs.get('boundaryMethod', self.CROP)
   
        
        return
       
        
       
    def initialise(self, img):
        
        if self.imSize < 0:
            if self.resize < 0:
                self.imSize = min(img.shape)
            else:
                self.imSize = self.resize

        if self.cropSize < 0:
            self.cropSize = round(self.imSize * .9)            
        
        if self.templateSize < 0:
            self.templateSize = round(self.imSize / 4)
            
        if self.refSize < 0:
            self.refSize = round(self.imSize / 2)
            
        if self.imageType == -1:
            self.imageType = img.dtype
        
        if np.size(self.mask) == 0:
            self.mask = PyBundle.getMask(np.zeros([self.imSize,self.imSize]),(self.imSize/2,self.imSize/2,self.cropSize / 2))
       
        self.mosaic = np.zeros((self.mosaicSize, self.mosaicSize), dtype = self.imageType)

        return 
    
    
    # Add image to current mosaic
    def add(self, img):

        # Before we have first image, can't choose sensilble default values
        if self.nImages == 0:
            self.initialise(img) 

                   
        if self.resize > 0:  #-1 means no resize
            imgResized = cv.resize(img, (self.resize, self.resize))
        else:
            imgResized = img
            
        if self.nImages > 0:
            self.lastShift = Mosaic.findShift(self.prevImg, imgResized, self.templateSize, self.refSize)
            self.currentX = self.currentX + self.lastShift[1]
            self.currentY = self.currentY + self.lastShift[0]
            
            distMoved = math.sqrt( (self.currentX - self.lastXAdded)**2 + (self.currentY - self.lastYAdded)**2)
            if distMoved >= self.minDistForAdd:
                self.lastXAdded = self.currentX
                self.lastYAdded = self.currentY
                
                for i in range(2):
                    outside, direction, outsideBy = Mosaic.isOutSideMosaic(self.mosaic, imgResized, (self.currentX, self.currentY))
    
                    if outside == True:
                        if self.boundaryMethod == self.EXPAND: 
                            self.mosaic, self.mosaicWidth, self.mosaicHeight, self.currentX, self.currentY = Mosaic.expandMosaic(self.mosaic, max(outsideBy, self.expandStep), direction, self.currentX, self.currentY)
                            outside = False
                        elif self.boundaryMethod == self.SCROLL:
                            self.mosaic, self.currentX, self.currentY = Mosaic.scrollMosaic(self.mosaic, outsideBy, direction, self.currentX, self.currentY)
                            outside = False
                    
 
                if outside == False:
                    if self.blend:
                        Mosaic.insertIntoMosaicBlended(self.mosaic, imgResized, self.mask, self.blendMask, self.cropSize, self.blendDist, (self.currentX, self.currentY))
                    else:
                        Mosaic.insertIntoMosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
                            
        else:  
            # 1st image goes straight into mosaic
            Mosaic.insertIntoMosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
            self.lastXAdded = self.currentX
            self.lastYAdded = self.currentY

        self.prevImg = imgResized
        self.nImages = self.nImages + 1




    # Return mosaic image
    def getMosaic(self):
        return self.mosaic
        

            
    # Dead leaf insertion of image into a mosaic at position. Only pixels for
    # which mask == 1 are copied
    def insertIntoMosaic(mosaic, img, mask, position):
        
        px = math.floor(position[0] - np.shape(img)[0] / 2)
        py = math.floor(position[1] - np.shape(img)[1] / 2)
        
        
        oldRegion = mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]]
        oldRegion[np.array(mask)] = img[np.array(mask)]
        mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]] = oldRegion
        return
    
    
    
    
    # Insertion of image into a mosaic with cosine window blending. Only pixels from
    # image for which mask == 1 are copied. Pixels within blendDist of edge of mosaic
    # (i.e. radius of cropSize/2) are blended with existing mosaic pixel values
    def insertIntoMosaicBlended(mosaic, img, mask, blendMask, cropSize, blendDist, position):
        
        
        px = math.floor(position[0] - np.shape(img)[0] / 2)
        py = math.floor(position[1] - np.shape(img)[1] / 2)
        
               
        # Region of mosaic we are going to work on
        oldRegion = mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]]

        # Only do blending on non-zero valued pixels of mosaic, otherwise we 
        # blend into nothing
        blendingMask = np.where(oldRegion>0,1,0)

        # If first time, create blend mask giving weights to apply for each pixel
        if blendMask == []:
            maskRad = cropSize / 2
            blendImageMask = Mosaic.cosineWindow(np.shape(oldRegion)[0], maskRad, blendDist) 


        imgMask = blendImageMask.copy()
        imgMask[blendingMask == 0] = 1   # For pixels where mosaic == 0 use original pixel values from image 
        imgMask = imgMask * mask
        mosaicMask = 1- imgMask          

        # Modify region to include blended values from image
        oldRegion = oldRegion * mosaicMask + img * imgMask
       

        # Insert it back in
        mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]] = oldRegion
       
        return
    
      
    
   
    
    # Calculates how far img2 has shifted relative to img1
    def findShift(img1, img2, templateSize, refSize):
        
        if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
            return -1
        else:
            template = PyBundle.extractCentral(img2, templateSize)  
            refIm = PyBundle.extractCentral(img1, refSize)
            res = cv.matchTemplate(template, refIm, cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            shift = [max_loc[0] - (refSize - templateSize), max_loc[1] - (refSize - templateSize)]
            return shift
                
    
    # Extracts sqaure of size boxSize from centre of img
    def extractCentral(img, boxSize):
        w = np.shape(img)[0]
        h = np.shape(img)[1]

        cx = w/2
        cy = h/2
        boxSemiSize = min(cx,cy,boxSize)
        
        img = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
        return img
        
    # Produce a circular cosine window mask on grid of imgSize * imgSize. Mask
    # is 0 for radius > circleSize and 1 for radius < (circleSize - cicleSmooth)
    # The intermediate region is a smooth cosine function.
    def cosineWindow(imgSize, circleSize, circleSmooth):
        
        innerRad = circleSize - circleSmooth
        xM, yM = np.meshgrid(range(imgSize),range(imgSize))
        imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
        mask =  np.cos(math.pi / (2 * circleSmooth) * (imgRad - innerRad))**2
        mask[imgRad < innerRad ] = 1
        mask[imgRad > innerRad + circleSmooth] = 0
        return mask
    
    # Checks if position of image to insert into mosaic will result in 
    # part of inserted images being outside of mosaic
    def isOutSideMosaic(mosaic, img, position):
        imgW = np.shape(img)[0] 
        imgH = np.shape(img)[1] 
        
        mosaicW = np.shape(mosaic)[0]
        mosaicH = np.shape(mosaic)[1]
        
                
        left = math.floor(position[0] - imgW / 2)
        top = math.floor(position[1] - imgH / 2)
        
        right = left + imgW 
        bottom = top + imgH 
        
        if left < 0 :
            return True, Mosaic.LEFT, -left
        elif top < 0:
            return True, Mosaic.TOP, -top
        elif right > mosaicW:
            return True, Mosaic.RIGHT, right - mosaicW        
        elif bottom > mosaicH:
            return True, Mosaic.BOTTOM, bottom - mosaicH
        else:
            return False, -1, 0
     
    # Increase size of mosaic image by 'distance' in direction 'direction'. Supply
    # currentX and currentY position so that these can be modified to be correct
    # for new mosaic size
    def expandMosaic(mosaic, distance, direction, currentX, currentY):
        mosaicWidth = np.shape(mosaic)[0]
        mosaicHeight = np.shape(mosaic)[1]

        if direction == Mosaic.LEFT:
            newMosaicWidth = mosaicWidth + distance
            newMosaic = np.zeros((newMosaicWidth, mosaicHeight), mosaic.dtype)
            newMosaic[distance:distance + mosaicWidth,:] = mosaic
            return newMosaic, newMosaicWidth, mosaicHeight, currentX + distance, currentY
             
        if direction == Mosaic.TOP:
            newMosaicHeight = mosaicHeight + distance
            newMosaic = np.zeros((mosaicWidth, newMosaicHeight), mosaic.dtype)
            newMosaic[:,distance:distance + mosaicHeight] = mosaic
            return newMosaic, mosaicWidth, newMosaicHeight, currentX,  currentY + distance 
        
        if direction == Mosaic.RIGHT:
            newMosaicWidth = mosaicWidth + distance
            newMosaic = np.zeros((newMosaicWidth, mosaicHeight), mosaic.dtype)
            newMosaic[0: mosaicWidth,:] = mosaic
            return newMosaic, newMosaicWidth, mosaicHeight, currentX, currentY
        
        if direction == Mosaic.BOTTOM:
            newMosaicHeight = mosaicHeight + distance
            newMosaic = np.zeros((mosaicWidth, newMosaicHeight), mosaic.dtype)
            newMosaic[:, 0:mosaicHeight ] = mosaic
            return newMosaic, mosaicWidth, newMosaicHeight,  currentX , currentY 
        
        
        
    # Scroll mosaic to allow mosaicing to continue past edge of mosaic. Pixel 
    # values will be lost. Supply currentX and currentY position so that these
    # can be modified to be correct for new mosaic size
    def scrollMosaic(mosaic, distance, direction, currentX, currentY):
        mosaicWidth = np.shape(mosaic)[0]
        mosaicHeight = np.shape(mosaic)[1]

        if direction == Mosaic.LEFT:       
            newMosaic = np.roll(mosaic,distance,0)
            newMosaic[0:distance:,:] = 0 
            return newMosaic, currentX + distance, currentY
             
        if direction == Mosaic.TOP:
            newMosaic = np.roll(mosaic,distance,1)
            newMosaic[:, 0:distance] = 0 
            return newMosaic, currentX,  currentY + distance 
        
        if direction == Mosaic.RIGHT:
            newMosaic = np.roll(mosaic,-distance,0)
            newMosaic[-distance:,:] = 0 
            return newMosaic, currentX  - distance, currentY
        
        if direction == Mosaic.BOTTOM:
            newMosaic = np.roll(mosaic,-distance,1)
            newMosaic[:, -distance:] = 0 
            return newMosaic, currentX,  currentY   - distance
        
        
        
        
        
        


