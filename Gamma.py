import ee
ee.Initialize()
import numpy as np
import math


def toNatural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    return ee.Image(img).log10().multiply(10.0)

def GammaMap(image, enl, ksize):
#Returns GammaMap filtered image

#Square kernel, ksize should be odd (typically 3, 5 or 7)
    weights = ee.List.repeat(ee.List.repeat(1,ksize),ksize)
### ~~(ksize/2) does integer division in JavaScript
    kernel = ee.Kernel.fixed(ksize,ksize, weights, int(ksize/2), int(ksize/2), False)
##Get mean and variance
    mean = image.reduceNeighborhood(ee.Reducer.mean(), kernel)
    variance = image.reduceNeighborhood(ee.Reducer.variance(), kernel)
## "Pure speckle" threshold
    ci = variance.sqrt().divide(mean)  # square root of inverse of enl
## If ci <= cu, the kernel lies in a "pure speckle" area -> return simple mean
    cu = 1.0/math.sqrt(enl)
## If cu < ci < cmax the kernel lies in the low textured speckle area -> return the filtered value
    cmax = math.sqrt(2.0) * cu

    alpha = ee.Image(1.0 + cu*cu).divide(ci.multiply(ci).subtract(cu*cu))
    b = alpha.subtract(enl + 1.0)
    d = mean.multiply(mean).multiply(b).multiply(b).add(alpha.multiply(mean).multiply(image).multiply(4.0*enl))
    f = b.multiply(mean).add(d.sqrt()).divide(alpha.multiply(2.0))
    combi = mean.updateMask(ci.lte(cu)).addBands(f.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))).addBands(image.updateMask(ci.gte(cmax)))
#If ci > cmax do not filter at all (i.e. we don't do anything, other then masking)
#Compose an single band image combining the mean filtered "pure speckle", the "low textured" filtered and the unfiltered portions
    return combi.reduce(ee.Reducer.sum())



def applyGamma(image):
#Return a GammaMap filtered version of the image converted to natural units
    filtered_VV = GammaMap((image.select('VV')), 5, 5)#.rename('gamma_VV');
    filtered_VH = GammaMap((image.select('VH')), 5, 5)#.rename('gamma_VH')
    return image.addBands(filtered_VV).addBands(filtered_VH).copyProperties(image)



def addSARIndex(image):
    NRPB = image.expression('(VH - VV) / (VH + VV)', { ## backscatter VH and VV polarization in dB
     'VV' : image.select(['gamma_VV']),
     'VH' : image.select(['gamma_VH'])}).rename(['NRPB'])
    return image.addBands(NRPB)###; //Normalized Ratio Rrocedure between Bands

def addSARRatio(image):
    ratio = image.expression(
    '(VH / VV)', { ###// backscatter VH and VV polarization in 
     'VV' : image.select(['gamma_VV']),
     'VH' : image.select(['gamma_VH'])}).rename(['Ratio'])
    return image.addBands(ratio); ###//Normalized Ratio Rrocedure between Bands

