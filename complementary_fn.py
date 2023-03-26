###*functions to obtain the local maximo/minimo values*

import numpy as np
import pandas as pd
import ee
def local_peak_general(ts):
# Find local peaks
# ts is a Series
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    df=pd.DataFrame(ts)
    df.rename(columns = {0:'data'}, inplace = True)
    # Find local peaks
    df['min'] = df.data[(df.data.shift(1) > df.data) & (df.data.shift(-1) > df.data)]
    df['max'] = df.data[(df.data.shift(1) < df.data) & (df.data.shift(-1) < df.data)]

    return df

def local_peak_potencial(ts,n):
# Find local peaks
# ts is a Series
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import argrelextrema
    df=pd.DataFrame(ts)
    df.rename(columns = {0:'data'}, inplace = True)
    #n  # number of points to be checked before and after
    df['mini'] = df.iloc[argrelextrema(df.data.values, np.less_equal,order=n)[0]]['data']
    df['maxi'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,order=n)[0]]['data']
    df['points']=df['mini'].fillna(df['maxi'])
    return df

def slope(x, win_size=5):
    av_derivative = np.empty(x.shape)
    av_derivative.fill(np.nan)
    #for i in range(x.shape[0]):
    half_window = int(win_size / 2)
    ln = x.shape[0]
    for i in range(half_window, ln - half_window):
        av_derivative[i] = np.nanmean(x[i - half_window : i]) - np.nanmean(x[i : i + half_window])
    return av_derivative



# Some modified from here: https://code.earthengine.google.com/2ef38463ebaf5ae133a478f173fd0ab5 [Originally by Guido Lemoine]
def toNatural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    return ee.Image(img).log10().multiply(10.0)

def RefinedLee(img):
    '''
    Refined Lee Speckle Filter
    NOTE: img must be in natural units, i.e. not in dB!
    '''
    #Set up 3x3 kernels 
    weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
    kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    #Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
    sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

    sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

    #Calculate mean and variance for the sampled windows and store as 9 bands
    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    #Determine the 4 gradients for the sampled windows
    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    #And find the maximum gradient amongst gradient bands
    max_gradient = gradients.reduce(ee.Reducer.max())

    #Create a mask for band pixels that are the maximum gradient
    gradmask = gradients.eq(max_gradient)

    #duplicate gradmask bands: each gradient represents 2 directions
    gradmask = gradmask.addBands(gradmask)

    #Determine the 8 directions
    directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
    directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
    directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
    directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
  
    #The next 4 are the not() of the previous 4
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    #Mask all values that are not 1-8
    directions = directions.updateMask(gradmask)

    #"collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
    directions = directions.reduce(ee.Reducer.sum()) 

    #var pal = ['ffffff','ff0000','ffff00', '00ff00', '00ffff', '0000ff', 'ff00ff', '000000'];
    #Map.addLayer(directions.reduce(ee.Reducer.sum()), {min:1, max:8, palette: pal}, 'Directions', false);

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

    #Calculate localNoiseVariance
    sigmaV = ee.Image(sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]))

    #Set up the 7*7 kernels for directional statistics
    rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))
    
    diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]])

    rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

    #Create stacks for mean and variance using the original kernels. Mask with relevant direction.
    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    #and add the bands for rotated kernels
    #for (var i=1; i<4; i++) {
    for i in range(1,4):
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

    #"collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    #And finally generate the filtered value
    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))
    b = varX.divide(dir_var)

    result = ee.Image(dir_mean.add(b.multiply(img.subtract(dir_mean))))
    return result

def apply_speckle_filt(collection):
    bn = collection.first().bandNames().getInfo()
    def applyfx(image):
        for b in bn:
            nat = toNatural(image.select(b)) #Convert to log scale
            filt = RefinedLee(nat) #Speckle Filter
            updated = toDB(filt) #Convert back to decibels
            image = image.addBands(updated.rename(b + '_filt'))
        return ee.Image(image)
    return collection.map(applyfx)



