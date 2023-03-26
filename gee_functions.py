# -*- coding: utf-8 -*-
"""
Functions used in the processes of the data for the thesis. 
Almost all the functions belong to Dr.Taylor Smith (@author: tsmith)
"""
import ee
ee.Initialize()
import numpy as np

################################################%% Utilities

########### For Time Series
def run_export(image, crs, filename, scale, region, maxPixels=1e12, cloud_optimized=True):
    '''
    Runs an export function on GEE servers
    '''
    task_config = {'fileNamePrefix': filename,'crs': crs,'scale': scale,'maxPixels': maxPixels, 'fileFormat': 'GeoTIFF', 'formatOptions': {'cloudOptimized': cloud_optimized}, 'region': region,}
    task = ee.batch.Export.image.toDrive(image, filename, **task_config)
    task.start()

def addtime(image):
    date = ee.Date(image.get('system:time_start'))
    years = date.difference(ee.Date('2017-01-01'), 'day')  #cambie aca la fecha
    return image.addBands(ee.Image(years).rename('t')).float().addBands(ee.Image.constant(1)) #Note we also add a constant here!

def percentile_reg(collection, band_name, percentiles):
    trend_images = []
    for p in percentiles:
        bands = ['constant', 't', band_name + '_p%i' % p]
        fit = collection.select(bands).reduce(ee.Reducer.robustLinearRegression(numX=2, numY=1))
        #fit = collection.select(bands).reduce(ee.Reducer.linearRegression(numX=2, numY=1))
        #fit = collection.select(bands).reduce(ee.Reducer.linearFit())
        #Note -- linearRegression and robustLinearRegression both also have 'residuals' to help you assess the fits
        lrImage = fit.select(['coefficients']).arrayProject([0]).arrayFlatten([['constant', 'trend']])\
            .select('trend')
        trend_images.append(lrImage)
    output_image = ee.ImageCollection.fromImages(trend_images).toBands().rename(['p_' + str(x) for x in percentiles])
    return output_image

def reduce_pct(collection, percentile, date_start, date_end, timeframe='month', skip=1):
    def reduce_pctile(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        agg = filt_coll.reduce(ee.Reducer.percentile(percentile)).set('system:time_start', t)
        return agg
    
    pctile = aggregate_to(collection, date_start, date_end, timeframe=timeframe, skip=skip, agg_fx=reduce_pctile)
    return pctile


###############%% Conversion to Python Time Series
def export_to_pandas(collection, clipper, aggregation_scale, med='median', save_std=True):
    '''
    Takes an ImageCollection, an Earth Engine Geometry, and an aggregation scale (e.g., 30m for Landsat, 250m for MODIS, etc)
    
    Returns a pandas time series for the mean/median and standard deviation values over the 
    aggregation area. 
    
    Optionally saves those time series to a CSV file    
    
    '''
    import pandas as pd, numpy as np
    
    def createTS(image):
        date = image.get('system:time_start')
        if med == 'median':
            value = image.reduceRegion(ee.Reducer.median(), clipper, aggregation_scale)
        elif med == 'mean':
            value = image.reduceRegion(ee.Reducer.mean(), clipper, aggregation_scale)
        else:
            value = image.reduceRegion(med, clipper, aggregation_scale)
        if save_std:
            std = image.reduceRegion(ee.Reducer.stdDev(), clipper, aggregation_scale)
            ft = ee.Feature(None, {'system:time_start': date, 'date': ee.Date(date).format('Y/M/d'), 'Mn': value, 'STD': std})
        else:
            ft = ee.Feature(None, {'system:time_start': date, 'date': ee.Date(date).format('Y/M/d'), 'Mn': value})
        return ft
    
    TS = collection.filterBounds(clipper).map(createTS)
    dump = TS.getInfo()
    fts = dump['features']
    out_vals = np.empty((len(fts)))
    out_dates = []
    out_std = np.empty((len(fts)))
    
    for i, f in enumerate(fts):
        props = f['properties']
        date = props['date']
        try:
            val = list(props['Mn'].values())[0]
        except:
            val = np.nan
        out_vals[i] = val
        
        if save_std:
            try:
                std = list(props['STD'].values())[0]
            except:
                std = np.nan
            out_std[i] = std
        out_dates.append(pd.Timestamp(date))
    
    ser = pd.Series(out_vals, index=out_dates)
    if save_std:
        serstd = pd.Series(out_std, index=out_dates)
        return ser, serstd
    else:
        return ser
    
def percentile_export(collection, percentile, clipper, aggregation_scale=30, save=None):
    '''
    Get a time series at a certain percentile
    '''
    
    import pandas as pd, numpy as np
    
    def createTS(image):
        date = image.get('system:time_start')
        value = image.reduceRegion(ee.Reducer.percentile([percentile]), clipper, aggregation_scale)
        ft = ee.Feature(None, {'system:time_start': date, 'date': ee.Date(date).format('Y/M/d'), 'PCT': value})
        return ft
    
    TS = collection.filterBounds(clipper).map(createTS)
    dump = TS.getInfo()
    fts = dump['features']
    out_vals = np.empty((len(fts)))
    out_dates = []
    
    for i, f in enumerate(fts):
        props = f['properties']
        date = props['date']
        try:
            val = list(props['PCT'].values())[0]
        except:
            val = np.nan
        out_vals[i] = val
        out_dates.append(pd.Timestamp(date))
    
    ser = pd.Series(out_vals, index=out_dates)
    if save:
        df = pd.DataFrame({'p' + str(percentile):out_vals, 'time':out_dates})
        df.to_csv(save + '.csv', index=False)
        print(save)
    return ser


def maskBorder(image):
    totalSlices = ee.Number(image.get('totalSlices'))
    sliceNumber = ee.Number(image.get('sliceNumber'))
    middleSlice = ee.Image(sliceNumber.gt(1).And(sliceNumber.lt(totalSlices)))
    mask = image.select(['VV', 'VH']).mask().reduce(ee.Reducer.min()).floor()
    pixelsToMask = mask.Not().fastDistanceTransform(128, 'pixels').sqrt()
    metersToMask = pixelsToMask.multiply(ee.Image.pixelArea().sqrt()).rename('metersToMask')
    notBorder = metersToMask.gte(500).And(pixelsToMask.gt(2))
    angle = image.select('angle')
    return image.updateMask(angle.gt(31).And(angle.lt(45)).And(middleSlice.Or(notBorder)))    

############################# shapefiles
def gee_geometry_from_shapely(geom, crs='epsg:4326'):
    """ 
    Simple helper function to take a shapely geometry and a coordinate system and convert them to a 
    Google Earth Engine Geometry.
    """
    from shapely.geometry import mapping
    ty = geom.type
    if ty == 'Polygon':
        return ee.Geometry.Polygon(ee.List(mapping(geom)['coordinates']), proj=crs, evenOdd=False)
    elif ty == 'Point':
        return ee.Geometry.Point(ee.List(mapping(geom)['coordinates']), proj=crs, evenOdd=False)    
    elif ty == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(ee.List(mapping(geom)['coordinates']), proj=crs, evenOdd=False)
    

###################################### Sentinel-2
def NDVI_S2(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').set('system:time_start', image.get('system:time_start'))
    return image.addBands(ndvi)

def NDWI_S2(image):
    ndwi = image.normalizedDifference(['B8', 'B11']).rename('NDWI').set('system:time_start', image.get('system:time_start'))
    return image.addBands(ndwi)

def MNDWI_S2(image):
    ndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI').set('system:time_start', image.get('system:time_start'))
    return image.addBands(ndwi)

def maskS2clouds(image):
    qa = image.select('QA60')
    
    #Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = (1 << 10)
    cirrusBitMask = (1 << 11)
    
    #Both flags should be set to zero, indicating clear conditions.
    mask =qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))## 
    return image.updateMask(mask).divide(10000).set('system:time_start', image.get('system:time_start'))

#%% Time aggregation
def aggregate_to(collection, ds, de, timeframe='month', skip=1, agg_fx='sum', agg_var=None):
    '''
    Take an ImageCollection and convert it into an aggregated value based on an arbitrary function.
    Several useful functions are included with keywords (mean, sum, etc), but an arbitrary reducer can be supplied
    
    day, month, year are the typical arguments
    
    skip will allow you to make larger windows (e.g., 5 days)
    
    '''
    start, end = ee.Date(ds), ee.Date(de)
    #Generate length of time steps to look through
    difdate = end.difference(start, timeframe)
    length = ee.List.sequence(0, difdate.subtract(1))
    
    if not skip == 1:
        length_py = length.getInfo()
        length_skip = length_py[::skip]
        length = ee.List(length_skip)
    
    def gen_datelist(t):
        return start.advance(t, timeframe)
    
    dates = length.map(gen_datelist)

    #Get band name
    bn = collection.first().bandNames().getInfo()[0]
    
    def create_sub_collections(t):
        t = ee.Date(t)
        filt_coll = collection.filterDate(t, t.advance(skip, timeframe)) #Move forward the 'skip' amount of time units
        return filt_coll.set('bandcount', ee.Number(filt_coll.size())).set('system:time_start', t.millis())
    
    mc = dates.map(create_sub_collections)
    mc_filt = mc.filter(ee.Filter.gt('bandcount',0))

    def reduceSum(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        daysum = filt_coll.reduce(ee.Reducer.sum()).set('system:time_start', t).rename(bn)
        return daysum
    
    def reduceMean(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        daymn = filt_coll.reduce(ee.Reducer.mean()).set('system:time_start', t).rename(bn)
        return daymn
    
    def reduceMin(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        daymn = filt_coll.reduce(ee.Reducer.min()).set('system:time_start', t).rename(bn)
        return daymn
    
    def reduceMax(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        daymn = filt_coll.reduce(ee.Reducer.max()).set('system:time_start', t).rename(bn)
        return daymn

    def reduceMedian(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        daymn = filt_coll.reduce(ee.Reducer.median()).set('system:time_start', t).rename(bn)
        return daymn
    
    def reduceSTD(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        daymn = filt_coll.reduce(ee.Reducer.stdDev()).set('system:time_start', t).rename(bn)
        return daymn
    
    def reduceIQR(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)
        t = filt_coll.get('system:time_start')
        pcts = filt_coll.reduce(ee.Reducer.percentile([25,75]))
        iqr = pcts.select(bn + '_p75').subtract(pcts.select(bn + '_p25')).toFloat().set('system:time_start', t).rename(bn)
        return iqr
    
    def reduce9010(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)  
        t = filt_coll.get('system:time_start')
        pcts = filt_coll.reduce(ee.Reducer.percentile([10,90]))
        iqr = pcts.select(bn + '_p90').subtract(pcts.select(bn + '_p10')).toFloat().set('system:time_start', t).rename(bn)
        return iqr
    
    def reduce955(filt_coll):
        filt_coll = ee.ImageCollection(filt_coll)  
        pcts = filt_coll.reduce(ee.Reducer.percentile([5,95]))
        t = filt_coll.get('system:time_start')
        iqr = pcts.select(bn + '_p95').subtract(pcts.select(bn + '_p5')).toFloat().set('system:time_start', t).rename(bn)
        return iqr
  
        rm = filt_coll.map(reclass)
        occur = rm.reduce(ee.Reducer.sum()).toUint8().set('system:time_start', t).rename(bn)
        return occur
    
    #Map over the list of months, return either a mean or a sum of those values
    if agg_fx == 'sum':
        mo_agg = mc_filt.map(reduceSum)
    elif agg_fx == 'min':
        mo_agg = mc_filt.map(reduceMin)
    elif agg_fx == 'max':
        mo_agg = mc_filt.map(reduceMax)
    elif agg_fx == 'mean':
        mo_agg = mc_filt.map(reduceMean)
    elif agg_fx == 'median':
        mo_agg = mc_filt.map(reduceMedian)
    elif agg_fx == 'std':
        mo_agg = mc_filt.map(reduceSTD)
    elif agg_fx == 'iqr':
        mo_agg = mc_filt.map(reduceIQR)
    elif agg_fx == '9010':
        mo_agg = mc_filt.map(reduce9010)
    else:
        mo_agg = mc_filt.map(agg_fx)
        
    #Convert back into an image collection
    agged = ee.ImageCollection.fromImages(mo_agg)
    
    return agged

def join_timeagg_collections(c1, c2):
    ''' Takes two collections with matching dates (e.g., time aggregated) and joins them into a 
    single collection '''
    filt = ee.Filter.equals(leftField='system:time_start', rightField='system:time_start') 
    #filt = ee.Filter.equals(leftField='system:index', rightField='system:index') #This only works if the collections are sorted in the same way!
    innerJoin = ee.Join.inner() #initialize the join
    innerJoined = innerJoin.apply(c1, c2, filt) #This is a FEATURE COLLECTION
    def combine_joined(feature):
        #Now we can go through the features and actually create new images from the features
        return ee.Image.cat(feature.get('primary'), feature.get('secondary'))
    
    joined_collect = ee.ImageCollection(innerJoined.map(combine_joined))
    return joined_collect




#################################################

from Gamma import *
from complementary_fn import *
################################################%% Utilities

#%% Sentinel 1 Specific Functions
def slope_correction(collection, model, buffer=0):
    #Via https://github.com/ESA-PhiLab/radiometric-slope-correction
    '''This function applies the slope correction on a collection of Sentinel-1 data
       
       :param collection: ee.Collection of Sentinel-1
       :param elevation: ee.Image of DEM
       :param model: model to be applied (volume/surface)
       :param buffer: buffer in meters for layover/shadow amsk
        
        :returns: ee.Image
    '''
    
    elevation = ee.Image('USGS/SRTMGL1_003')
    
    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        '''Code for calculation of volumetric model SCF
        
        :param theta_iRad: ee.Image of incidence angle in radians
        :param alpha_rRad: ee.Image of slope steepness in range
        
        :returns: ee.Image
        '''
        
        # create a 90 degree image in radians
        ninetyRad = ee.Image.constant(90).multiply(np.pi/180)
        
        # model
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator) 
    
    
    def _surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        '''Code for calculation of direct model SCF
        
        :param theta_iRad: ee.Image of incidence angle in radians
        :param alpha_rRad: ee.Image of slope steepness in range
        :param alpha_azRad: ee.Image of slope steepness in azimuth
        
        :returns: ee.Image
        '''
        
        # create a 90 degree image in radians
        ninetyRad = ee.Image.constant(90).multiply(np.pi/180)
        
        # model  
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = (alpha_azRad.cos()
          .multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos()))

        return nominator.divide(denominator)


    def _erode(image, distance):
      '''Buffer function for raster

      :param image: ee.Image that shoudl be buffered
      :param distance: distance of buffer in meters
        
      :returns: ee.Image
      '''
      
      d = (image.Not().unmask(1)
          .fastDistanceTransform(30).sqrt()
          .multiply(ee.Image.pixelArea().sqrt()))
    
      return image.updateMask(d.gt(distance))
    
    
    def _masking(alpha_rRad, theta_iRad, buffer):
        '''Masking of layover and shadow
        
        
        :param alpha_rRad: ee.Image of slope steepness in range
        :param theta_iRad: ee.Image of incidence angle in radians
        :param buffer: buffer in meters
        
        :returns: ee.Image
        '''
        # layover, where slope > radar viewing angle 
        layover = alpha_rRad.lt(theta_iRad).rename('layover')

        # shadow 
        ninetyRad = ee.Image.constant(90).multiply(np.pi/180)
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
        
        # add buffer to layover and shadow
        if buffer > 0:
            layover = _erode(layover, buffer)   
            shadow = _erode(shadow, buffer)  

        # combine layover and shadow
        no_data_mask = layover.And(shadow).rename('no_data_mask')
        
        return layover.addBands(shadow).addBands(no_data_mask)
                        
        
    def _correct(image):
        '''This function applies the slope correction and adds layover and shadow masks
        
        '''
        
        # get the image geometry and projection
        geom = image.geometry()
        proj = image.select(1).projection()
        
        # calculate the look direction
        heading = (ee.Terrain.aspect(image.select('angle'))
                                     .reduceRegion(ee.Reducer.mean(), geom, 1000)
                                     .get('aspect'))
                   

        # Sigma0 to Power of input image
        sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

        # the numbering follows the article chapters
        # 2.1.1 Radar geometry 
        theta_iRad = image.select('angle').multiply(np.pi/180)
        phi_iRad = ee.Image.constant(heading).multiply(np.pi/180)
        
        # 2.1.2 Terrain geometry
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(np.pi/180).setDefaultProjection(proj).clip(geom)
        phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(np.pi/180).setDefaultProjection(proj).clip(geom)
        
        # we get the height, for export 
        height = elevation.setDefaultProjection(proj).clip(geom)
        
        # 2.1.3 Model geometry
        #reduce to 3 angle
        phi_rRad = phi_iRad.subtract(phi_sRad)

        # slope steepness in range (eq. 2)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        # slope steepness in azimuth (eq 3)
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

        # local incidence angle (eq. 4)
        theta_liaRad = (alpha_azRad.cos().multiply((theta_iRad.subtract(alpha_rRad)).cos())).acos()
        theta_liaDeg = theta_liaRad.multiply(180/np.pi)

        # 2.2 
        # Gamma_nought
        gamma0 = sigma0Pow.divide(theta_iRad.cos())
        gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(['VV', 'VH'], ['gamma_VV', 'gamma_VH'])
        ratio_gamma = (gamma0dB.select('gamma_VV')
                        .subtract(gamma0dB.select('gamma_VH'))
                        .rename('ratio_gamma0'))

        if model == 'volume':
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)

        if model == 'surface':
            scf = _surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        # apply model for Gamm0_f
        gamma0_flat = gamma0.divide(scf)
        gamma0_flatDB = (ee.Image.constant(10)
                         .multiply(gamma0_flat.log10())
                         .select(['VV', 'VH'],['VV_gamma0flat', 'VH_gamma0flat'])
                        )

        masks = _masking(alpha_rRad, theta_iRad, buffer)

        # calculate the ratio for RGB vis
        ratio_flat = (gamma0_flatDB.select('VV_gamma0flat')
                        .subtract(gamma0_flatDB.select('VH_gamma0flat'))
                        .rename('ratio_gamma0flat')
                     )
#'VV_sigma0', 'VH_sigma0'
        return (image.rename(['VV', 'VH', 'incAngle','VV_gamma_sigma0', 'VH_gamma_sigma0'])
                      .addBands(gamma0dB)
                      .addBands(ratio_gamma)
                      .addBands(gamma0_flatDB)
                      .addBands(ratio_flat)
                      .addBands(alpha_rRad.rename('alpha_rRad'))
                      .addBands(alpha_azRad.rename('alpha_azRad'))
                      .addBands(phi_sRad.rename('aspect'))
                      .addBands(alpha_sRad.rename('slope'))
                      .addBands(theta_iRad.rename('theta_iRad'))
                      .addBands(theta_liaRad.rename('theta_liaRad'))
                      .addBands(masks)
                      .addBands(height.rename('elevation'))
                 )    
    
    # run and return correction
    return collection.map(_correct)

def fix_S1(ds, de, polygon, orbit=False, direction='Ascending'):
    S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    S1 = S1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filterBounds(polygon)\
        .filterDate(ds, de)\
        .map(maskBorder)\
        .map(applyGamma)\
        .map(terrainCorrection)
   ###     .map(addSARIndex)\
   ###     .map(addSARRatio)\
    ###    .select(['gamma_VH', 'gamma_VV','NRPB', 'Ratio'])

    #####
    
    if orbit==True:
        S1 = S1.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit))
    
    if direction == 'Ascending':
        data = S1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    else:
        data = S1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    
    #Apply angle masking
    data = data.map(maskAngGT30)
    data = data.map(maskAngLT45)
    
#     #Apply terrain correction
#     if gamma:
#         data = slope_correction(data, 'surface', buffer=0)
#         #Choose the gamma bands and rename
#         def rename(collection, which):
#             def rnfx(image):
#                 return image.rename(['VV', 'VH']).set('system:time_start', image.get('system:time_start'))
#             return collection.select(which).map(rnfx)            
                
        #data = rename(data, ['gamma_VV', 'gamma_VH'])
    
#     s1_crs = data.select('VV').first().projection()
    
    return data

def filter_s1(Ascending):
    def make_rat(image):
        rat = image.select('VV').divide(image.select('VH'))
        return rat.rename('VVdVH').set('system:time_start', image.get('system:time_start'))
    
    def make_rat_filt(image):
        rat = image.select('VV_filt').divide(image.select('VH_filt'))
        return rat.rename('VVdVH').set('system:time_start', image.get('system:time_start'))
    
    def make_dif(image):
        rat = image.select('VV').subtract(image.select('VH'))
        return rat.rename('VVminVH').set('system:time_start', image.get('system:time_start'))
                                       
    S1A_both = Ascending.select(['VV', 'VH']).sort('system:time_start')
    S1A_both_filt = apply_speckle_filt(S1A_both)
    
    S1A_both_focal = focal_med_filt(S1A_both)
    S1A_ratio_focal = S1A_both_focal.map(make_rat_filt)
    S1A_ratio_focal = mask_invalid(S1A_ratio_focal, -5, 5)
        
    S1A_ratio = S1A_both.map(make_rat)
    S1A_ratio_filt = S1A_both_filt.map(make_rat_filt)
    S1A_ratio_filt = mask_invalid(S1A_ratio_filt, -5, 5)
    S1A_dif = S1A_both.map(make_dif)
    
    return S1A_both, S1A_both_focal, S1A_both_filt, S1A_ratio, S1A_ratio_filt, S1A_ratio_focal

#Translate to Gamma0
def make_gamma0(image):
    angle = image.select('angle').resample('bicubic')
    return image.select('..').divide(angle.multiply(np.pi/180.0).cos()).copyProperties(image, ['system:time_start'])

#Edge masking with high/low angle
def maskAngGT30(image):
    ang = image.select(['angle'])
    return image.updateMask(ang.gt(30.63993))

def maskAngLT45(image):
    ang = image.select(['angle'])
    return image.updateMask(ang.lt(45.53993)) 

def maskAngleGT40(image):
    ang = image.select(['angle'])
    return image.updateMask(ang.gt(40))

###################################################
def terrainCorrection(image):
    import numpy as np
    #Implementation by Andreas Vollrath (ESA), inspired by Johannes Reiche (Wageningen)
    #Modified from: https://gis.stackexchange.com/questions/352602/getting-local-incidence-angle-from-sentinel-1-grd-image-collection-in-google-ear
    imgGeom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom) # 30m srtm 
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

    #Article ( numbers relate to chapters)
    #2.1.1 Radar geometry 
    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i)\
        .reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000)\
        .get('aspect')

    #2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')

    #2.1.3 Model geometry
    #reduce to 3 angle
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)

    #convert all to radians
    phi_rRad = phi_r.multiply(np.pi / 180)
    alpha_sRad = alpha_s.multiply(np.pi / 180)
    theta_iRad = theta_i.multiply(np.pi / 180)
    ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)

    #slope steepness in range (eq. 2)
    alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

    #slope steepness in azimuth (eq 3)
    alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

    #local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
    theta_liaDeg = theta_lia.multiply(180 / np.pi)
    
    #2.2 Gamma_nought_flat
    gamma0 = sigma0Pow.divide(theta_iRad.cos())
    gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'))

    #Volumetric Model
    nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
    denominator = (ninetyRad.subtract(theta_iRad)).tan()
    volModel = (nominator.divide(denominator)).abs()

    #apply model
    gamma0_Volume = gamma0.divide(volModel)
    gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

    #we add a layover/shadow maskto the original implmentation
    #layover, where slope > radar viewing angle 
    alpha_rDeg = alpha_r.multiply(180 / np.pi)
    layover = alpha_rDeg.lt(theta_i)

    #shadow where LIA > 90
    shadow = theta_liaDeg.lt(85)

    #calculate the ratio for RGB vis
    ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'))

    output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad)\
    .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

#     return gamma0dB
    return image.addBands(output.select(['VV', 'VH', 'slope_1', 'slope_2'], ['VV', 'VH', 'layover', 'shadow']),None, True)


#%% Online Filtering
def apply_SG(collect, window_size=7, imageAxis=0, bandAxis=1, order=3, ds='2018-09-01'):
    '''
    Apply a Savitzky-Golay filter to a time series collection (pixelwise)
 geom,
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
    
#     sg = sg_series.map(minmax)   
# sg.filterMetadata('roi_min', 'not_equals', None).filterMetadata('roi_max', 'not_equals', None)
    
    return sg_series