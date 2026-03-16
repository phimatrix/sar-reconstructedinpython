"""
This script performs GEE-based SAR-Optical fusion.
"""

# ==============================
# --- Proxy (if required) ---
# ==============================

import os

os.environ['HTTP_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'
os.environ['HTTPS_PROXY'] = 'http://rrsceast:NRSC%40User@192.168.0.9:8080'

# ==============================
# --- Imports ---
# ==============================

import json
import time
import ee
import GEE_funcs
import utilities
from utilities import check_task_status


# ==============================
# --- Earth Engine Authentication ---
# ==============================

cred_path = os.path.expanduser("~/.config/earthengine/credentials")

if not os.path.exists(cred_path):
    print("No GEE credentials found. Authenticating now...")
    ee.Authenticate()
else:
    print("GEE credentials found. Using existing credentials.")


# ==============================
# --- Initialize Earth Engine ---
# ==============================

PROJECT_ID = "theta-arcana-484116-h7"

ee.Initialize(project=PROJECT_ID)

print(f"Earth Engine initialized successfully with project: {PROJECT_ID}")


# ==============================
# --- Load Parameters ---
# ==============================

PARAM_FILE = "Parameters.json"

if not os.path.exists(PARAM_FILE):
    raise FileNotFoundError(f"{PARAM_FILE} not found!")

with open(PARAM_FILE, 'r') as f:
    cfg = json.load(f)

AOI_PATH = cfg.get(
    "AOI_PATH",
    "projects/theta-arcana-484116-h7/assets/grid_123"
)

print(f"AOI_PATH: {AOI_PATH}")

PROJECT_TITLE = cfg.get("PROJECT_TITLE", "GEE_Project")
OPTICAL_MISSION = cfg.get("OPTICAL_MISSION", "S2")
START_DATE = cfg.get("START_DATE", "2023-01-01")
END_DATE = cfg.get("END_DATE", "2023-12-31")

PCA_SMOOTH = cfg.get("PCA_SMOOTH", True)
PCA_COMPONENT_RATIO = cfg.get("PCA_COMPONENT_RATIO", 0.9)
STD_CLOUD_THRESHOLD = cfg.get("STD_CLOUD_THRESHOLD", 30)


# ==============================
# --- Run Fusion For AOI ---
# ==============================

def run_for_aoi(aoi_path, aoi_id):

    AOI = ee.FeatureCollection(aoi_path).geometry()

    print("Running fusion for:", aoi_id)

    # ---------------------------
    # Manage GEE Assets
    # ---------------------------

    parent_folder = f"projects/{PROJECT_ID}/assets"
    subasset_folder = f"{parent_folder}/{PROJECT_TITLE}"

    assets = ee.data.listAssets({'parent': parent_folder}).get('assets', [])
    asset_names = [asset['name'].split('/')[-1] for asset in assets]

    if PROJECT_TITLE not in asset_names:
        ee.data.createAsset({'type': 'Folder'}, subasset_folder)

    # ---------------------------
    # Optical Collection
    # ---------------------------

    optical_collection = ee.ImageCollection({
        'L8': "LANDSAT/LC08/C01/T1_SR",
        'S2': "COPERNICUS/S2_SR_HARMONIZED"
    }[OPTICAL_MISSION]) \
        .filterBounds(AOI) \
        .filterDate(START_DATE, END_DATE)

    optical_collection = GEE_funcs.prepare_optical(
        optical_collection,
        AOI,
        OPTICAL_MISSION
    )

    optical_collection = optical_collection.filterMetadata(
        'PIXEL_COUNT_AOI',
        'greater_than',
        100
    )

    # ---------------------------
    # SAR Collection
    # ---------------------------

    S1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(AOI) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH'])

    # ---------------------------
    # Pairing
    # ---------------------------

    indep_variables = [
        'VV_mean', 'VH_mean', 'VV_diff', 'VH_diff',
        'VV_Nmean', 'VH_Nmean', 'VV_Ndiff', 'VH_Ndiff'
    ]

    opt_SAR = GEE_funcs.pair_opt_SAR(
        optical_collection,
        S1,
        AOI,
        indep_variables
    )

    pair_count = opt_SAR.size().getInfo()

    if pair_count < 1:
        raise ValueError('No image pairs found.')

    print(f'Image pairs found: {pair_count}')

    # ---------------------------
    # Add constant band
    # ---------------------------

    opt_SAR = opt_SAR.map(
        lambda img: img.addBands(
            img.select(0).multiply(0).add(1).rename('constant')
        )
    )

    opt_SAR_train = opt_SAR.select(
        ['constant'] + indep_variables + ['NDVI']
    )

    # ---------------------------
    # Robust Linear Regression
    # ---------------------------

    robust_linear_regression = opt_SAR_train.reduce(
        ee.Reducer.robustLinearRegression(
            numX=len(indep_variables) + 1,
            numY=1
        )
    )

    rlr_image = robust_linear_regression.select(
        ['coefficients']
    ).arrayFlatten(
        [['constant'] + indep_variables, ['NDVI']]
    )

    # ---------------------------
    # Prediction
    # ---------------------------

    def MLR_predict(img):

        NDVI_pred = img.select(
            ['constant'] + indep_variables
        ).multiply(
            rlr_image.rename(['constant'] + indep_variables)
        ).reduce('sum').rename('NDVI_pred')

        return img.addBands(NDVI_pred)

    opt_SAR_outputs = opt_SAR.map(MLR_predict)

    # ---------------------------
    # PCA Smoothing
    # ---------------------------

    if PCA_SMOOTH:

        NDVI_smoothed = GEE_funcs.Temporal_PCA(
            opt_SAR_outputs.select('NDVI_pred'),
            AOI,
            opt_SAR_outputs.size()
            .multiply(PCA_COMPONENT_RATIO)
            .floor()
            .int(),
            50
        )

    else:

        NDVI_smoothed = opt_SAR_outputs.select(
            'NDVI_pred'
        ).toBands()

    NDVI_calibrated, NDVI_filled = GEE_funcs.post_process(
        opt_SAR_outputs,
        NDVI_smoothed,
        AOI,
        STD_CLOUD_THRESHOLD
    )

    # ---------------------------
    # Export
    # ---------------------------

    OutputFilled = NDVI_filled.select(
        'NDVI'
    ).toBands().clip(AOI)

    utilities.export_image_todrive(
        OutputFilled,
        AOI,
        f"Reconstructed_NDVI_{aoi_id}",
        PROJECT_TITLE
    )

    print("Export started for:", aoi_id)


# ==============================
# --- Loop Through All AOIs ---
# ==============================

for i in range(201, 251):

    aoi_id = f"grid_{i}"

    AOI_PATH = f"projects/{PROJECT_ID}/assets/{aoi_id}"

    print("Processing:", AOI_PATH)

    try:

        run_for_aoi(AOI_PATH, aoi_id)

        time.sleep(20)

    except Exception as e:

        print("Failed:", AOI_PATH, e)
