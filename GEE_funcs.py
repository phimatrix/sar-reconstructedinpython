import ee
import time
import os

# ⚠️ Only keep proxy if absolutely required
# os.environ['HTTP_PROXY'] = 'http://username:password@ip:port'
# os.environ['HTTPS_PROXY'] = 'http://username:password@ip:port'

# Do NOT initialize here — fusion.py handles it
# ee.Initialize(project='theta-arcana-484116-h7')

import os

os.environ['HTTP_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'
os.environ['HTTPS_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'


def prepare_optical(optical_collection, AOI, optical_mission):
    """
     Preprocess optical images
    """

    def cal_NDVI(img):
        """Step1 : Calculate NDVI for optical images"""

        if optical_mission == 'S2':
            nir = img.select('B8')
            red = img.select('B4')
        else:  # Landsat 8
            nir = img.select('B5')
            red = img.select('B4')

        NDVI = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return img.addBands(NDVI)

    def add_cloudMask_L8(img):
        """ Step2: Process Landsat 8 cloud flag
        In the cloud mask, 1 is for cloudy pixel.
        """

        cirrus_bitMask = 1 << 2
        cloudshadow_bitMask = 1 << 3
        clouds_bitMask = 1 << 5

        qa = img.select('pixel_qa')

        mask = qa.bitwiseAnd(cloudshadow_bitMask) \
            .Or(qa.bitwiseAnd(clouds_bitMask)) \
            .Or(qa.bitwiseAnd(cirrus_bitMask)).rename('Mask')

        cloud_sum = ee.Number(
            mask.reduceRegion(
                ee.Reducer.sum(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))

        cloud_count = ee.Number(
            mask.reduceRegion(
                ee.Reducer.count(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))

        cloud_percent = cloud_sum.divide(cloud_count).multiply(100)

        return img.addBands(mask).set({
            'CLOUD_PERCENTAGE_AOI': cloud_percent,
            'PIXEL_COUNT_AOI': cloud_count})

    def add_cloudMask_S2(img):
        """ Step2: Process Sentinel cloud flag
        Number 3 is cloud shadow and above 7 are cloud with different
        confidence respectively.
        In the cloud mask, 1 is for cloudy pixel.
        """

        scl = img.select('SCL')

        mask = scl.eq(3).Or(scl.gte(8)).rename('Mask')

        cloud_sum = ee.Number(
            mask.reduceRegion(
                ee.Reducer.sum(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))

        cloud_count = ee.Number(
            mask.reduceRegion(
                ee.Reducer.count(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))

        cloud_percent = cloud_sum.divide(cloud_count).multiply(100)

        return img.addBands(mask).set({
            'CLOUD_PERCENTAGE_AOI': cloud_percent,
            'PIXEL_COUNT_AOI': cloud_count})

    if optical_mission == 'L8':
        optical_collection = optical_collection.map(cal_NDVI).map(
            add_cloudMask_L8).select(['NDVI', 'Mask'])
    else:
        optical_collection = optical_collection.map(cal_NDVI).map(
            add_cloudMask_S2).select(['NDVI', 'Mask'])

    return optical_collection


def cal_covariates(img, AOI, indep_variables):
    """
    Step 3 Calculate temporal and spatial covariates with spatial-first method
    """

    img = img.updateMask(img.gt(-40))

    spatial_mean = img.select(['VV', 'VH']) \
        .reduceRegion(ee.Reducer.mean(), AOI, 100, maxPixels=1e9) \
        .toImage(['VV', 'VH'])

    neighbor_mean = img.select(['VV', 'VH']) \
        .reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(10))

    spatial_diff = img.select(['VV', 'VH']) \
        .subtract(spatial_mean.select(['VV', 'VH']))

    neighbor_diff = img.select(['VV', 'VH']) \
        .subtract(ee.Image(neighbor_mean.select(['VV_mean', 'VH_mean'])))

    img = img.addBands(spatial_mean.select(['VV', 'VH'])
                       .rename(['VV_mean', 'VH_mean'])) \
        .addBands(ee.Image(neighbor_mean.select(['VV_mean', 'VH_mean']))
                  .rename(['VV_Nmean', 'VH_Nmean'])) \
        .addBands(spatial_diff.select(['VV', 'VH'])
                  .rename(['VV_diff', 'VH_diff'])) \
        .addBands(neighbor_diff.select(['VV', 'VH'])
                  .rename(['VV_Ndiff', 'VH_Ndiff']))

    img = img.select(indep_variables).focal_median().toFloat()

    return img


def pair_opt_SAR(optical_collection, SAR_collection, AOI, indep_variables):
    """
    Step 5 Pair optical and SAR image collections
    """

    def pair_image(img):

        S2_date = ee.Date(img.get('system:time_start'))

        S1_filtered = SAR_collection.filterDate(
            S2_date.advance(-12, 'day'),
            S2_date.advance(12, 'day'))

        S1_composite = S1_filtered.mean()

        S1_covariates = cal_covariates(
            S1_composite, AOI, indep_variables)

        img = img.set({'S1_COUNT': S1_filtered.size()})

        img = ee.Algorithms.If(
            S1_filtered.size().gt(0),
            img.addBands(S1_covariates),
            img)

        return ee.Image(img)

    opt_SAR = optical_collection.map(pair_image).filterMetadata(
        'S1_COUNT', 'greater_than', 0)

    return opt_SAR


def Temporal_PCA(imageCollection, AOI, numComponent=10, scale=10):
    """
    Apply pixelwise PCA analysis along each time series
    """

    image = imageCollection.toBands().clip(AOI)
    band_names = image.bandNames()

    mean_dict = image.reduceRegion(
        ee.Reducer.mean(), AOI, scale, maxPixels=1e9)

    means = ee.Image.constant(mean_dict.values(band_names))
    centered = image.subtract(means)

    arrays = centered.toArray()

    covar = arrays.reduceRegion(
        ee.Reducer.centeredCovariance(),
        AOI,
        scale,
        maxPixels=1e9)

    covar_array = ee.Array(covar.get('array'))
    eigens = covar_array.eigen()

    eigen_values = eigens.slice(1, 0, 1)
    eigen_vectors = eigens.slice(1, 1)

    array_image = arrays.toArray(1)
    principal_components = ee.Image(eigen_vectors).matrixMultiply(array_image)

    eigen_slice = eigen_vectors.slice(0, 0, numComponent)
    pc_slice = principal_components.arraySlice(0, 0, numComponent)

    pc_inverse = ee.Image(eigen_slice.transpose()).matrixMultiply(pc_slice)

    smooth_results = pc_inverse.arrayProject([0]).arrayFlatten(
        [band_names]).add(means)

    return smooth_results


def post_process(paired_collection, prediction,
                 AOI, std_cloud_threshold):
    """
    Calibrate NDVI predictions and fill gaps
    """

    def pred_standardize(img):

        img_id = img.id().cat('_NDVI_pred')
        mask = img.select('Mask')
        cloud_cover = ee.Number(img.get('CLOUD_PERCENTAGE_AOI'))

        NDVI_pred = prediction.select(img_id).rename('NDVI')
        NDVI_obs = img.select('NDVI')

        reducer = ee.Reducer.mean().combine(
            ee.Reducer.intervalMean(85, 95), '90', True).combine(
            ee.Reducer.intervalMean(5, 15), '10', True)

        pred_stats = NDVI_pred.updateMask(
            mask.eq(0)).reduceRegion(
            reducer, AOI, 100, maxPixels=1e9)

        obs_stats = NDVI_obs.updateMask(
            mask.eq(0)).reduceRegion(
            reducer, AOI, 100, maxPixels=1e9)

        pred_90mean = ee.Number(pred_stats.get('NDVI_90mean'))
        pred_10mean = ee.Number(pred_stats.get('NDVI_10mean'))
        pred_mean = ee.Number(pred_stats.get('NDVI_mean'))

        pred_uprange = pred_90mean.subtract(pred_mean)
        pred_downrange = pred_10mean.subtract(pred_mean)

        obs_90mean = ee.Number(obs_stats.get('NDVI_90mean'))
        obs_10mean = ee.Number(obs_stats.get('NDVI_10mean'))
        obs_mean = ee.Number(obs_stats.get('NDVI_mean'))

        obs_uprange = obs_90mean.subtract(obs_mean)
        obs_downrange = obs_10mean.subtract(obs_mean)

        pred_uprange = ee.Number(
            ee.Algorithms.If(pred_uprange.neq(0), pred_uprange, obs_uprange))

        pred_downrange = ee.Number(
            ee.Algorithms.If(pred_downrange.neq(0), pred_downrange, obs_downrange))

        NDVI_diff = NDVI_pred.subtract(pred_mean)

        NDVI_upcalibrated = NDVI_diff.divide(pred_uprange) \
            .multiply(obs_uprange).add(obs_mean).rename('NDVI_pred')

        NDVI_downcalibrated = NDVI_diff.divide(pred_downrange) \
            .multiply(obs_downrange).add(obs_mean).rename('NDVI_pred')

        NDVI_calibrated = NDVI_upcalibrated.where(
            NDVI_diff.lt(0), NDVI_downcalibrated)

        NDVI_calibrated = ee.Algorithms.If(
            cloud_cover.lt(std_cloud_threshold),
            NDVI_calibrated,
            NDVI_pred)

        return img.addBands(
            ee.Image(NDVI_calibrated).rename('NDVI_pred'),
            overwrite=True)

    NDVI_calibrated = paired_collection.map(pred_standardize)

    def gap_infilling(img):

        prediction = img.select('NDVI_pred')
        observation = img.select('NDVI')
        cloudMask = img.select('Mask')

        gap_filled = observation.where(cloudMask, prediction).focal_median()

        error = observation.subtract(
            prediction).abs().updateMask(cloudMask.eq(0))

        MAE = ee.Number(
            error.reduceRegion(
                ee.Reducer.mean(),
                AOI,
                100,
                maxPixels=1e9).get('NDVI'))

        MAE = ee.Algorithms.If(MAE, MAE, -1)

        return ee.Image(gap_filled).addBands(cloudMask).toFloat().copyProperties(
            img,
            ['random', 'Split_label', 'CLOUD_PERCENTAGE_AOI',
             'system:time_start', 'system:index']
        ).set({'MAE': MAE})

    NDVI_filled = NDVI_calibrated.map(gap_infilling)

    return NDVI_calibrated, NDVI_filled
