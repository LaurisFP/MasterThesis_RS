def fix_time_SPOT(t):
    import datetime
    t_fix = datetime.datetime(int(t[:4]), int(t[4:6]), int(t[6:]))
    return t_fix

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Takes input of series, window_size, order, deriv, and rate
    DEFAULT: savitzky_golay(values, 21, 1, deriv=0, rate=1)
    window_size must be ODD
    """
    from math import factorial
    import numpy as np
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def chen_proc(orig_input):
    import itertools
    import numpy as np, pandas as pd
    """Following Chen et al. 2004"""
    #Remove impossible values, linear fit
    orig_input[orig_input.diff() > 0.4] = np.nan
    
    #Remove remaining missed clouds/water
    orig_input[orig_input < 0] = np.nan
    
    #Remove series with too few values
    if np.nansum(np.isnan(orig_input)) > 300: 
        return False, False
        
    #Mask out non-vegetated areas (de Jong et al.)
#     mns = [] 
#     for yr in orig_input.groupby(orig_input.index.year):
#         yrdata = yr[1]
#         mns.append(np.nanmean(yrdata.values))
    
#     if np.nanmean(mns) < 0.1: 
#         return False, False
    
    #STEP 1 - Do the interpolation    
    else:
        data = orig_input.interpolate('linear').bfill().ffill()
        N_0 = data.copy()
        
        #STEP 2 - Fit SG Filter, using best fitting parameters
        arr = np.empty((4,3))
        for m,d in list(itertools.product([4,5,6,7],[2,3,4])):
            data_smoothed = pd.Series(savitzky_golay(data.values, 2*m + 1, d, deriv=0, rate=1), index=data.index)
            err = np.nansum(data_smoothed - data)**2
            arr[m-4,d-2] = err
        m,d = np.where(arr == np.nanmin(arr))
        m,d = m[0]+4, d[0]+2
        data_smoothed = pd.Series(savitzky_golay(data.values, 2*m + 1, d, deriv=0, rate=1), index=data.index)
        diff = N_0 - data_smoothed
        
        #STEP 3 - Create weights array based on difference from curve
        max_dif = np.nanmax(np.abs(diff.values))
        weights = np.zeros(np.array(data_smoothed.values).shape)
        weights[data_smoothed >= N_0] = 1
        weights[data_smoothed < N_0] = 1 - (np.abs(diff.values)/max_dif)[data_smoothed < N_0]
        
        #STEP 4 - Replace values that were smoothed downwards with original values
        data_smoothed[N_0 >= data_smoothed] = N_0[N_0 >= data_smoothed]
        data_smoothed[N_0 < data_smoothed] = data_smoothed[N_0 < data_smoothed]
        
        #STEP 5 - Resmooth with different parameters
        data_fixed = pd.Series(savitzky_golay(data_smoothed.values, 9, 6, deriv=0, rate=1), index=data.index)
        
        #STEP 6 - Calculate the fitting effect 
        Fe_0 = np.nansum(np.abs(data_fixed.values - N_0.values) * weights)            
        
        data = data_fixed.copy()
        
        count = 0
        #while Fe_0 >= Fe and Fe <= Fe_1:
        #while Fe_0 >= Fe:
        while count < 5:
            #STEP 4 - Replace values that were smoothed downwards with original values
            data[N_0 >= data] = N_0[N_0 >= data]
            data[N_0 < data] = data[N_0 < data]
            
            #STEP 5 - Resmooth with different parameters
            data_fixed = pd.Series(savitzky_golay(data.values, 9, 6, deriv=0, rate=1), index=data.index)
            #data_fixed = pd.Series(savitzky_golay(data.values, 5, 2, deriv=0, rate=1), index=data.index)

            #STEP 6 - Calculate the fitting effect 
            Fe = np.nansum(np.abs(data_fixed.values - N_0.values) * weights)
            data = data_fixed.copy()
            Fe_0 = Fe.copy()
            count += 1
            #if Fe <= Fe_0:
                
                #data_fixed = data_fixed_smoothed.copy()
                #data_fixed[N_0 >= data_fixed_smoothed] = N_0[N_0 >= data_fixed_smoothed]
                #data_fixed[N_0 < data_fixed_smoothed] = data_fixed_smoothed[N_0 < data_fixed_smoothed]
            #if Fe > Fe_0:
            #    break
        return np.array(data.values), True
    
    
    
    
###########################################
def apply_SG(collect, window_size=7, imageAxis=0, bandAxis=1, order=3):
    '''
    Apply a Savitzky-Golay filter to a time series collection (pixelwise), geom

    '''
    def prep_SG(img):
        #Add predictors for SG fitting, using date difference
        #We prepare for order 3 fitting, but can be adapted to lower order fitting later on
        dstamp = ee.Date(img.get('system:time_start'))
        ddiff = dstamp.difference(ee.Date(ds), 'hour')
        return img.addBands(ee.Image(1).toFloat().rename('constant'))\
        .addBands(ee.Image(ddiff).toFloat().rename('t'))\
        .addBands(ee.Image(ddiff).pow(ee.Image(2)).toFloat().rename('t2'))\
        .addBands(ee.Image(ddiff).pow(ee.Image(3)).toFloat().rename('t3'))\
        .set('date', dstamp)
    
    def getLocalFit(i):
        #Get a slice corresponding to the window_size of the SG smoother
        subarray = array.arraySlice(imageAxis, ee.Number(i).int(), ee.Number(i).add(window_size).int())
        #predictors = subarray.arraySlice(bandAxis, 2, 2 + order + 1)
        predictors = subarray.arraySlice(bandAxis, 1, 1 + order + 1) #Here for a one-variable case
        response = subarray.arraySlice(bandAxis, 0, 1)
        coeff = predictors.matrixSolve(response)
        
        coeff = coeff.arrayProject([0]).arrayFlatten(coeffFlattener)
        return coeff 
    
    def apply_SG_sub(i):
        ref = ee.Image(c.get(ee.Number(i).add(ee.Number(half_window))))
        return getLocalFit(i).multiply(ref.select(indepSelectors)).reduce(ee.Reducer.sum()).copyProperties(ref)
    
#     doy=collect.select('DOY')
    
    
    half_window = (window_size - 1)/2
    if order == 3:
        coeffFlattener = [['constant', 'x', 'x2', 'x3']]
        indepSelectors = ['constant', 't', 't2', 't3']
    elif order == 2:
        coeffFlattener = [['constant', 'x', 'x2']]
        indepSelectors = ['constant', 't', 't2']
        
    collect_coeffs = collect.map(prep_SG)
    array = collect_coeffs.toArray() #THIS STEP IS EXPENSIVE
    c = collect_coeffs.toList(collect_coeffs.size())
    runLength = ee.List.sequence(0, c.size().subtract(window_size))
    
    sg_series = runLength.map(apply_SG_sub)
    
    #Drop the null values
    sg_sliced = sg_series.slice(half_window, sg_series.size().subtract(half_window))
    sg_series = ee.ImageCollection.fromImages(sg_sliced)
    
#     def minmax(img):
#         #Map reducer to get global min/max NDVI (for filtering purposes)
#         bn = img.bandNames().get(0)
#         minMax =  img.reduceRegion(ee.Reducer.minMax(), geom, 1000)
#         return img.set({'roi_min': minMax.get(ee.String(bn).cat('_min')),'roi_max': minMax.get(ee.String(bn).cat('_max'))}).set('date', img.get('date'))
    
    sg = sg_series####.map(minmax)                    .filterMetadata('roi_min', 'not_equals', None).filterMetadata('roi_max', 'not_equals', None)
   
    return sg.addBands()