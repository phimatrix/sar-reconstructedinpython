import ee

# Correct: just the project number / ID
ee.Initialize(project='1020974277772')
print("GEE initialized!")

# Test AOI
aoi = ee.FeatureCollection("projects/1020974277772/assets/grid_372")
print("Number of features in AOI:", aoi.size().getInfo())
