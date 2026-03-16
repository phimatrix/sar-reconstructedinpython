import ee

# Correct: just the project number / ID
ee.Initialize(project='theta-arcana-484116-h7')

aoi = ee.FeatureCollection("projects/theta-arcana-484116-h7/assets/grid_210")
print("GEE initialized!")

# Test AOI

print("Number of features in AOI:", aoi.size().getInfo())
