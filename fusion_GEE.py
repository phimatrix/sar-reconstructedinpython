"""
This script performs GEE-based SAR-Optical fusion.
"""

import os
import json
import ee
import GEE_funcs
from utilities import check_task_status
import utilities


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

# ✅ USE YOUR CLOUD PROJECT ID (NOT numeric)
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

# ✅ Use your actual asset path
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
# --- Define AOI ---
# ==============================

AOI = ee.FeatureCollection(AOI_PATH).geometry()


def main():

    print(f"Project {PROJECT_TITLE} started.")
    nameSuffix = f"_{OPTICAL_MISSION}_{START_DATE}_{END_DATE}"

    # ==============================
    # --- Manage GEE Assets ---
    # ==============================

    parent_folder = f"projects/{PROJECT_ID}/assets"
    subasset_folder = f"{parent_folder}/{PROJECT_TITLE}"

    assets = ee.data.listAssets({'parent': parent_folder}).get('assets', [])
    asset_names = [asset['name'].split('/')[-1] for asset in assets]

    if PROJECT_TITLE not in asset_names:
        ee.data.createAsset({'type': 'Folder'}, subasset_folder)

    # ==============================
    # --- Preprocess Optical Images ---
    # ==============================

    optical_collection = ee.ImageCollection({
        'L8': "LANDSAT/LC08/C01/T1_SR",
        'S2': "COPERNICUS/S2_SR_HARMONIZED"
    }[OPTICAL_MISSION]) \
        .filterBounds(AOI) \
        .filterDate(START_DATE, END_DATE)

    optical_collection = GEE_funcs.prepare_optical(
        optical_collection, AOI, OPTICAL_MISSION
    )

    optical_collection = optical_collection.filterMetadata(
        'PIXEL_COUNT_AOI', 'greater_than', 100
    )

    # ==============================
    # --- Preprocess SAR Images ---
    # ==============================

    S1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(AOI) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH'])

    # ==============================
    # --- Pairing ---
    # ==============================

    indep_variables = [
        'VV_mean', 'VH_mean', 'VV_diff', 'VH_diff',
        'VV_Nmean', 'VH_Nmean', 'VV_Ndiff', 'VH_Ndiff'
    ]

    opt_SAR = GEE_funcs.pair_opt_SAR(
        optical_collection, S1, AOI, indep_variables
    )

    pair_count = opt_SAR.size().getInfo()

    if pair_count < 1:
        raise ValueError('No image pairs found.')
    else:
        print(f'Image pairs found: {pair_count}. Continuing regression...')

    # ADD CONSTANT BAND HERE
    opt_SAR = opt_SAR.map(
        lambda img: img.addBands(img.select(0).multiply(0).add(1).rename('constant')
        )
    )

    # ==============================
    # --- Robust Linear Regression ---
    # ==============================

    opt_SAR_train = opt_SAR.filterMetadata(
        'Split_label', 'not_equals', 'Testing'
    )

    opt_SAR_train = opt_SAR_train.map(
        lambda img: img.updateMask(img.select('Mask').eq(0))
    )
    opt_SAR_train = opt_SAR_train.select(
        ['constant'] + indep_variables + ['NDVI']
    )

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

    # ==============================
    # --- Prediction ---
    # ==============================

    def MLR_predict(img):
        NDVI_pred = img.select(
            ['constant'] + indep_variables
        ).multiply(
            rlr_image.rename(['constant'] + indep_variables)
        ).reduce('sum').rename('NDVI_pred')

        return img.select(['NDVI', 'Mask']).addBands(NDVI_pred)

    opt_SAR_outputs = opt_SAR.map(MLR_predict)

    # ==============================
    # --- PCA Smoothing ---
    # ==============================

    if PCA_SMOOTH:
        NDVI_smoothed = GEE_funcs.Temporal_PCA(
            opt_SAR_outputs.select('NDVI_pred'),
            AOI,
            opt_SAR_outputs.size()
                .multiply(PCA_COMPONENT_RATIO)
                .floor()
                .int(),
            10
        )
    else:
        NDVI_smoothed = opt_SAR_outputs.select('NDVI_pred').toBands()

    NDVI_calibrated, NDVI_filled = GEE_funcs.post_process(
        opt_SAR_outputs,
        NDVI_smoothed,
        AOI,
        STD_CLOUD_THRESHOLD
    )

    # ==============================
    # --- Export ---
    # ==============================

    OutputFilled = NDVI_filled.select('NDVI').toBands().clip(AOI)

    utilities.export_image_todrive(
        OutputFilled,
        AOI,
        "Reconstructed_NDVI_July2023",
        PROJECT_TITLE
    )

    print("Processing and export complete!")


if __name__ == "__main__":
    main()
